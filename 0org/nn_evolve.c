// nn_evolve.c
// OpenMPによる並列化対応 + 評価関数改善バージョン
// Compile: gcc -O3 -lm -fopenmp -o nn_evolve nn_evolve.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>

#define MAX_WAY 2000 // 増やしました
#define MAX_POP 200
#define MAX_SENS 5

double clamp(double x, double a, double b){ return x<a? a : (x>b? b : x); }
double randd(){ return (double)rand()/RAND_MAX; }
double uniform(double a,double b){ return a + (b-a)*randd(); }

typedef struct {
    int n;
    double x[MAX_WAY];
    double y[MAX_WAY];
    double half_width;
    double total_length; // コース全長を保持
} Track;

// 2点間の距離
double dist(double x1, double y1, double x2, double y2){
    return hypot(x2-x1, y2-y1);
}

int load_track_text(const char *fn, Track *t){
    FILE *f = fopen(fn,"r");
    if(!f) return 0;
    int n;
    if(fscanf(f,"%d %lf",&n,&t->half_width)!=2){ fclose(f); return 0;}
    if(n > MAX_WAY) n = MAX_WAY;
    t->n = n;
    for(int i=0;i<n;i++) fscanf(f,"%lf %lf",&t->x[i],&t->y[i]);
    fclose(f);
    
    // コース全長を計算（評価関数の正規化に使用）
    t->total_length = 0.0;
    for(int i=0; i<t->n; i++){
        int next = (i+1) % t->n; // 閉じたコースとして計算
        t->total_length += dist(t->x[i], t->y[i], t->x[next], t->y[next]);
    }
    return 1;
}

// 点(px,py)から線分(ax,ay)-(bx,by)への最短距離を返す
double point_to_seg_dist(double px,double py,double ax,double ay,double bx,double by){
    double vx = bx-ax, vy = by-ay;
    double wx = px-ax, wy = py-ay;
    double c = vx*vx+vy*vy;
    if(c==0) return hypot(px-ax, py-ay);
    double t = (vx*wx + vy*wy)/c;
    if(t<0) t=0; if(t>1) t=1;
    double qx = ax + t*vx, qy = ay + t*vy;
    return hypot(px-qx, py-qy);
}

typedef struct { double x,y,theta,v; } Car;

void car_step(Car *c, double steer_cmd, double throttle, double dt){
    double max_steer = 0.7;
    double steer = clamp(steer_cmd, -1.0, 1.0) * max_steer;
    double L = 0.5;
    // キネマティックバイシクルモデル
    c->theta += (c->v * tan(steer) / L) * dt;
    
    double max_acc = 5.0; // 少し加速性能を上げました
    double friction = 0.1; // 摩擦抵抗
    
    c->v += (clamp(throttle, -1.0, 1.0) * max_acc - c->v * friction) * dt;
    
    // 速度制限
    if(c->v < -2.0) c->v = -2.0;
    if(c->v > 12.0) c->v = 12.0;
    
    c->x += c->v * cos(c->theta) * dt;
    c->y += c->v * sin(c->theta) * dt;
}

void sense(Track *tr, Car *c, int ns, double angles[], double maxdist, double out[]){
    // センサー処理は重いので、少し最適化したいが今回はロジック維持
    // 本来は近傍探索を使うべき
    for(int i=0;i<ns;i++){
        double a = c->theta + angles[i];
        double dx = cos(a), dy = sin(a);
        double step = 2.0; 
        double d=0.0;
        int hit=0;
        
        // 簡易的なレイキャスト（精度より速度優先ならstepを大きく）
        for(; d<=maxdist; d+=step){
            double px = c->x + dx*d, py = c->y + dy*d;
            double mind = 1e9;
            
            // 全探索は重いが、とりあえずそのまま（必要なら近傍探索へ）
            // 高速化のためステップを飛ばしてチェック
            for(int s=0;s<tr->n;s+=2){ 
                int next = (s+1)%tr->n;
                double pd = point_to_seg_dist(px,py,tr->x[s],tr->y[s],tr->x[next],tr->y[next]);
                if(pd < mind) mind = pd;
                if(mind < tr->half_width) break; // 早期脱出
            }
            if(mind > tr->half_width){ hit=1; break; } // コース外判定
        }
        
        // コース内なら d は maxdist まで伸びる。コース外に出たらその距離。
        // ここでは「壁までの距離」ではなく「コース領域内にいるか」を判定しているロジックに見えるため
        // 元のロジック（mind > half_width で hit=1 つまり壁衝突）に従う
        
        if(!hit) d=maxdist; 
        out[i] = d / maxdist;
    }
}

typedef struct { int in,hid,out,n_w; double *w; } NN;
NN *nn_create(int in,int hid,int out){
    NN *nn=(NN*)malloc(sizeof(NN)); nn->in=in; nn->hid=hid; nn->out=out;
    nn->n_w = hid*in + hid + out*hid + out;
    nn->w=(double*)malloc(sizeof(double)*nn->n_w);
    return nn;
}
void nn_free(NN *nn){ free(nn->w); free(nn); }
void nn_set_weights(NN *nn, double *w){ memcpy(nn->w, w, sizeof(double)*nn->n_w); }
double tanh_act(double x){ return tanh(x); } // 高速化のため近似関数にしてもよい

void nn_forward(NN *nn, double in[], double out[]){
    int p=0;
    double hid[100]; // stack allocation enough for max hid
    for(int j=0;j<nn->hid;j++){
        double s=0.0;
        for(int i=0;i<nn->in;i++) s += nn->w[p++] * in[i];
        s += nn->w[p++];
        hid[j] = tanh_act(s);
    }
    for(int k=0;k<nn->out;k++){
        double s=0.0;
        for(int j=0;j<nn->hid;j++) s += nn->w[p++] * hid[j];
        s += nn->w[p++];
        out[k] = tanh_act(s);
    }
}

typedef struct { double *genes; double fitness; } Genome;
Genome *g_create(int n){ Genome *g=(Genome*)malloc(sizeof(Genome)); g->genes=(double*)malloc(sizeof(double)*n); return g; }
void g_free(Genome *g){ free(g->genes); free(g); }

// ★ 評価関数（大幅修正） ★
double eval(Genome *g, Track *tr, int steps, int ns, double *angs, double maxdist){
    NN *nn = nn_create(ns+1, 12, 2); // 隠れ層少し増強
    nn_set_weights(nn, g->genes);
    
    Car car; 
    car.x = tr->x[0]; car.y = tr->y[0];
    // 初期向き：0番目から1番目の点へのベクトル
    car.theta = atan2(tr->y[1]-tr->y[0], tr->x[1]-tr->x[0]);
    car.v = 0.0;
    
    double fitness = 0.0;
    int current_wp_idx = 0; // 現在の最寄りウェイポイントインデックス
    double dt = 0.05;

    for(int t=0; t<steps; t++){
        // --- センサー入力 ---
        double s[MAX_SENS], in[20], out[2];
        sense(tr, &car, ns, angs, maxdist, s);
        for(int i=0;i<ns;i++) in[i] = s[i];
        in[ns] = car.v / 12.0; // 速度の正規化
        
        // --- NN推論 & 移動 ---
        nn_forward(nn, in, out);
        double prev_x = car.x;
        double prev_y = car.y;
        car_step(&car, out[0], out[1], dt); // ステアリング, スロットル
        
        double dx = car.x - prev_x;
        double dy = car.y - prev_y;
        double move_dist = hypot(dx, dy);
        
        // --- 衝突判定（最寄りの線分との距離）---
        // 高速化：現在のウェイポイント周辺だけ探す
        int best_idx = current_wp_idx;
        double min_dist = 1e9;
        
        // 前後10点程度を探索（コース周回対応）
        for(int k = -5; k <= 10; k++){
            int idx = (current_wp_idx + k + tr->n) % tr->n;
            int next = (idx + 1) % tr->n;
            double d = point_to_seg_dist(car.x, car.y, tr->x[idx], tr->y[idx], tr->x[next], tr->y[next]);
            if(d < min_dist){
                min_dist = d;
                if(k > 0 && k < 5) best_idx = idx; // 少し進んだ位置ならインデックス更新候補
            }
        }
        current_wp_idx = best_idx; // インデックス更新
        
        if(min_dist > tr->half_width){
            // コースアウトしたらそこで終了（少しペナルティ）
            fitness -= 0.1; 
            break;
        }
        
        // --- ★ 完走率の変化量による重み付け計算 ★ ---
        if(move_dist > 0.001){
            // 今いるセグメントの方向ベクトル（接線）を取得
            int next_wp = (current_wp_idx + 1) % tr->n;
            double tx = tr->x[next_wp] - tr->x[current_wp_idx];
            double ty = tr->y[next_wp] - tr->y[current_wp_idx];
            double t_len = hypot(tx, ty);
            if(t_len > 0){ tx /= t_len; ty /= t_len; }
            
            // 車の移動ベクトルと、コース進行方向ベクトルの内積をとる
            // 正ならコース順行（反時計回り）、負なら逆走
            double forward_progress = dx * tx + dy * ty;
            
            // 「コース全長に対する進捗割合」を加算
            // これが「完走率の変化量で重み付け」に相当
            fitness += forward_progress / tr->total_length;
            
            // 逆走への強いペナルティを与えたい場合
            // if(forward_progress < 0) fitness += forward_progress * 2.0 / tr->total_length;
        }
        
        // 速度が出なさすぎる個体への微小ペナルティ（スタック防止）
        if(t > 50 && car.v < 0.1) fitness -= 0.001;
    }
    
    // 1周（1.0）以上回った場合のボーナスなどをつけてもよい
    // 現状は純粋に距離比率の累積
    
    nn_free(nn);
    return (fitness > 0) ? fitness : 0.0;
}

void mutate(Genome *g,int n,double mag){
    for(int i=0;i<n;i++) if(randd()<0.05) g->genes[i]+=uniform(-mag,mag);
}
void crossover(Genome *a, Genome *b, Genome *out, int n){
    int cp=rand()%n;
    for(int i=0;i<n;i++) out->genes[i]=(i<cp)?a->genes[i]:b->genes[i];
}
int cmp(const void *a, const void *b){
    double fa=(*(Genome**)a)->fitness, fb=(*(Genome**)b)->fitness;
    return (fa<fb)?1:(fa>fb)?-1:0;
}

int main(int argc, char **argv){
    srand((unsigned)time(NULL));
    if(argc < 2){ 
        printf("Usage: %s track.txt [num_threads]\n", argv[0]); 
        return 1; 
    }
    
    int num_threads = 20;
    if(argc >= 3) num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads > 0 ? num_threads : 4);

    Track tr;
    if(!load_track_text(argv[1],&tr)) { printf("Err: track load\n"); return 1; }
    printf("Track loaded: %d points, Total Length: %.2f\n", tr.n, tr.total_length);
    
    int GENS = 500;     
    int STEPS = 2000;  
    int NSENS = 5;
    // センサー角度（前方広め）
    double angs[] = {-1.2, -0.6, 0.0, 0.6, 1.2};
    double maxdist = 50.0;
    
    int nin=NSENS+1, nhid=12, nout=2;
    int nw = nhid*nin + nhid + nout*nhid + nout;
    
    Genome **pop = (Genome**)malloc(sizeof(Genome*)*MAX_POP);
    for(int i=0;i<MAX_POP;i++){
        pop[i] = g_create(nw);
        for(int k=0;k<nw;k++) pop[i]->genes[k]=uniform(-1.0,1.0);
    }
    
    Genome *next[MAX_POP];
    
    for(int g=0;g<GENS;g++){
        // 並列計算
        #pragma omp parallel for schedule(dynamic)
        for(int i=0;i<MAX_POP;i++){
            pop[i]->fitness = eval(pop[i],&tr,STEPS,NSENS,angs,maxdist);
        }
        
        qsort(pop,MAX_POP,sizeof(Genome*),cmp);
        
        if(g%10==0) printf("Gen %d Best Fitness: %.4f (%.1f%% Lap)\n", 
                           g, pop[0]->fitness, pop[0]->fitness * 100.0);
        
        // エリート保存
        int elit=20; 
        for(int i=0;i<elit;i++){
            next[i]=g_create(nw);
            memcpy(next[i]->genes,pop[i]->genes,sizeof(double)*nw);
        }
        
        // 選択・交叉・変異
        while(elit<MAX_POP){
            // トーナメント選択もどき
            int r1=rand()%MAX_POP; int r2=rand()%MAX_POP;
            Genome *p1 = (pop[r1]->fitness > pop[r2]->fitness) ? pop[r1] : pop[r2];
            
            r1=rand()%MAX_POP; r2=rand()%MAX_POP;
            Genome *p2 = (pop[r1]->fitness > pop[r2]->fitness) ? pop[r1] : pop[r2];
            
            Genome *child=g_create(nw);
            crossover(p1,p2,child,nw);
            
            // 世代が進むにつれ変異を少し弱める
            mutate(child,nw, 0.5 * (1.0 - 0.8*(double)g/GENS));
            next[elit++]=child;
        }
        
        for(int i=0;i<MAX_POP;i++) g_free(pop[i]);
        for(int i=0;i<MAX_POP;i++) pop[i]=next[i];
    }
    
    FILE *f=fopen("all_pop.weights","wb");
    if(f){ 
        for(int i=0; i<MAX_POP; i++){
            fwrite(pop[i]->genes,sizeof(double),nw,f);
        }
        fclose(f); 
    }
    printf("Done.\n");
    
    for(int i=0;i<MAX_POP;i++) g_free(pop[i]);
    free(pop);
    return 0;
}