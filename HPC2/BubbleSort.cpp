#include <omp.h>
#include <bits/stdc++.h>

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using namespace std;

void s_bubble(int* a, int n)
{
    for (int i = 0; i < n; i++)
    {
        int first = i % 2;
        for (int j = first; j < n - 1; j += 2)
        {
            if (a[j] > a[j + 1])
            {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

void p_bubble(int* a, int n)
{
    for (int i = 0; i < n; i++)
    {
        int first = i % 2;
#pragma omp parallel for shared(a, first) num_threads(2)
        for (int j = first; j < n - 1; j += 2)
        {
            if (a[j] > a[j + 1])
            {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

void swap(int& a, int& b)
{
    int temp;
    temp = a;
    a = b;
    b = temp;
}

int main(int argc, const char** argv)
{
 
    int* a, n;
    n = stoi(argv[1]);
    a = new int[n];

    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % n;
    }

    cout << "Generated random array of length " << n << "\n\n";

    auto start1 = high_resolution_clock::now();
    s_bubble(a, n);
    auto stop1 = high_resolution_clock::now();
    int sequentialTime = duration_cast<milliseconds>(stop1 - start1).count();


    auto start2 = high_resolution_clock::now();
    p_bubble(a, n);
    auto stop2 = high_resolution_clock::now();
    int parallelTime = duration_cast<milliseconds>(stop2 - start2).count();

    float speedUp = (float)sequentialTime / parallelTime;
    float efficiency = speedUp / 2;

    cout<<endl;
    cout << "Sequential Bubble sort: " << sequentialTime << "ms\n";
    cout << "Parallel (2) Bubble sort: " << parallelTime << "ms\n";
    cout << "Speed Up: " << speedUp << "\n";
    cout << "Efficiency: " << efficiency << "\n";

    return 0;
}
