#include<bits/stdc++.h>
#include <omp.h>

using namespace std;

// Function to perform DFS from a given vertex
void dfs(int startVertex, vector<bool>& visited, vector<vector<int>>& graph)
{
    // Create a stack for DFS
    stack<int> s;

    // Mark the start vertex as visited and push it onto the stack
    visited[startVertex] = true;
    s.push(startVertex);

    // Loop until the stack is empty
    while (!s.empty())
    {
        // Pop a vertex from the stack
        int v = s.top();
        s.pop();

        // Push all adjacent vertices that are not visited onto the stack
        #pragma omp parallel for
        for (int i = 0; i < graph[v].size(); i++)
        {
            int u = graph[v][i];
            #pragma omp critical
            {
                if (!visited[u])
                {
                    visited[u] = true;
                    s.push(u);
                }
            }
        }
    }
}

// Parallel Depth-First Search
void dfsutil(vector<vector<int>>& graph, int numCores)
{
    int numVertices = graph.size();
    vector<bool> visited(numVertices, false); // Keep track of visited vertices
    double startTime = omp_get_wtime(); // Start timer

    // Perform DFS from all unvisited vertices using specified number of cores
    #pragma omp parallel for num_threads(numCores)
    for (int v = 0; v < numVertices; v++)
    {
        if (!visited[v])
        {
            dfs(v, visited, graph);
        }
    }

    double endTime = omp_get_wtime(); // End timer

    cout << "Time taken: " << endTime - startTime << " seconds" << endl;
    cout << "------------------------" << endl;
}

int main()
{
    // Generate a random graph with 10,000 vertices and 50,000 edges
    int numVertices = 10000;
    int numEdges = 10000;
    vector<vector<int>> graph(numVertices);
    srand(time(0));

    for (int i = 0; i < numEdges; i++)
    {
        int u = rand() % numVertices;
        int v = rand() % numVertices;
        graph[u].push_back(v);
        graph[v].push_back(u);
    }

    // Array containing number of cores

    cout << "Running Sequential DFS: " << endl;
    dfsutil(graph, 1);
    cout<<endl;
    cout << "Running Parallel DFS: " << endl;
    dfsutil(graph, 2);

    return 0;
}
