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

    double start1 = omp_get_wtime(); // Start timer
    s_bubble(a, n);
    double stop1 = omp_get_wtime(); // Stop timer
    double sequentialTime = stop1 - start1;


    double start2 = omp_get_wtime(); // Start timer
    p_bubble(a, n);
    double stop2 = omp_get_wtime(); // Stop timer
    double parallelTime= stop2 - start2;

    double speedUp = (float)sequentialTime / parallelTime;
    double efficiency = speedUp / 2;

    cout<<endl;
    cout << "Sequential Bubble sort: " << sequentialTime << " seconds\n";
    cout << "Parallel (2) Bubble sort: " << parallelTime << " seconds\n";
    cout << "Speed Up: " << speedUp << "\n";
    cout << "Efficiency: " << efficiency << "\n";

    return 0;
}
