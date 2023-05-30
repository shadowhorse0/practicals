#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

void s_avg(int arr[], int n)
{
    long sum = 0L;
    for (int i = 0; i < n; i++)
    {
        sum = sum + arr[i];
    }
    double avg = avg/n;
}

void p_avg(int arr[], int n)
{
    long sum = 0L;
    int i;
#pragma omp parallel for reduction(+ : sum) num_threads(16)
    for (i = 0; i < n; i++)
    {
        sum = sum + arr[i];
    }
    double avg = avg/n;
}

void s_sum(int arr[], int n)
{
    long sum = 0L;
    for (int i = 0; i < n; i++)
    {
        sum = sum + arr[i];
    }
 
}

void p_sum(int arr[], int n)
{
    long sum = 0L;
#pragma omp parallel for reduction(+ : sum) num_threads(16)
    for (int i = 0; i < n; i++)
    {
        sum = sum + arr[i];
    }
}

void s_max(int arr[], int n)
{
    int max_val = INT_MIN;
    for (int i = 0; i < n; i++)
    {
        if (arr[i] > max_val)
        {
            max_val = arr[i];
        }
    }
}

void p_max(int arr[], int n)
{
    int max_val = INT_MIN;
#pragma omp parallel for reduction(max : max_val) num_threads(16)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] > max_val)
        {
            max_val = arr[i];
        }
    }
}

void s_min(int arr[], int n)
{
    int min_val = INT_MAX;
    for (int i = 0; i < n; i++)
    {
        if (arr[i] < min_val)
        {
            min_val = arr[i];
        }
    }
}

void p_min(int arr[], int n)
{
    int min_val = INT_MAX;
#pragma omp parallel for reduction(min : min_val) num_threads(16)
    for (int i = 0; i < n; i++)
    {
        if (arr[i] < min_val)
        {
            min_val = arr[i];
        }
    }
}

int traverse(function<void()> fn)
{
    auto start = chrono::high_resolution_clock::now();
    fn();
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    return duration.count();
}

int main(int argc, const char **argv)
{
    int *a, n, i;
    n = stoi(argv[1]);
    a = new int[n];
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % n;
    }

    cout << "Generated random array of length: " << n << endl;
    omp_set_num_threads(16);

    int s_min_t = traverse([&] {
        s_min(a, n);
    });

    int p_min_t = traverse([&] {
        p_min(a, n);
    });

    int s_max_t = traverse([&] {
        s_max(a, n);
    });

    int p_max_t = traverse([&] {
        p_max(a, n);
    });

    int s_sum_t = traverse([&] {
        s_sum(a, n);
    });

    int p_sum_t = traverse([&] {
        p_sum(a, n);
    });

    int s_avg_t = traverse([&] {
        s_avg(a, n);
    });

    int p_avg_t = traverse([&] {
        p_avg(a, n);
    });

    cout << "Sequential Min: " << s_min_t << " ms"<<endl;
    cout << "Parallel Min: " << p_min_t<< " ms"<<endl;

    cout << "Sequential Max: " << s_max_t << " ms"<<endl;
    cout << "Parallel Max: " << p_max_t<< " ms"<<endl;

    cout << "Sequential Sum: " << s_sum_t<< " ms"<<endl;
    cout << "Parallel Sum: " << p_sum_t<< " ms"<<endl;

    cout << "Sequential Average: " << s_avg_t << " ms"<<endl;
    cout << "Parallel Average: " << p_avg_t << " ms"<<endl;

    return 0;
}
