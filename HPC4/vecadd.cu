#include<bits/stdc++.h>
using namespace std;

__global__
void vectorAdd(int* a, int* b, int* c, int size){
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size){
        c[index]=a[index]+b[index];
    }
}

void seqVecAdd(int *a, int* b, int* c,int size){
    for(int i=0; i<size; i++){
        c[i]=a[i]+b[i];
    }
}

int main(){
    int size = 10000;
    int* a = new int[size];
    int* b = new int[size];
    int* c = new int[size];

    for(int i=0; i<size; i++){
        a[i]=rand()%100;
        b[i]=rand()%100;
    }

    int *d_a,*d_b,*d_c;
    cudaMalloc((void**)&d_a,sizeof(int)*size);
    cudaMalloc((void**)&d_b,sizeof(int)*size);
    cudaMalloc((void**)&d_c,sizeof(int)*size);

    cudaMemcpy(d_a,a,sizeof(int)*size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,b,sizeof(int)*size,cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (size+threads-1)/threads;
    auto start1 = chrono::high_resolution_clock::now();
    vectorAdd<<<blocks,threads>>>(d_a,d_b,d_c,size);
    cudaDeviceSynchronize();
    auto stop1 = chrono::high_resolution_clock::now();

    cudaMemcpy(c,d_c,sizeof(int)*size,cudaMemcpyDeviceToHost);

    auto start2 = chrono::high_resolution_clock::now();
    seqVecAdd(a,b,c,size);
    auto stop2 = chrono::high_resolution_clock::now();

    auto p_time = chrono::duration_cast<chrono::microseconds>(stop1-start1);
    auto s_time = chrono::duration_cast<chrono::microseconds>(stop2-start2);

    cout<<"Sequential Algorithm time: "<<s_time.count()<<endl;
    cout<<"Parallel Algorithm time: "<<p_time.count()<<endl;

    for(int i=0; i<10; i++){
        cout<<a[i]<<" + "<<b[i]<<" = "<<c[i]<<endl;
    }
    
    delete[] a;
    delete[] b;
    delete[] c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}