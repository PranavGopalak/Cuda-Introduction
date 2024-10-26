// Simple MPI program to emulate what the Excel spreadsheet model does for:
//
// velocity(t) = Sum[(accel(t)]
//
// position(t) = Sum[velocity(t)]
//
// Using as many ranks as you wish - should ideally divide into 1800 to avoid possible residual errors.
//
// Due to loop carried dependencies, this requires basic dynamic programming to use the final condition of one
// rank as the initial condition of others at a later time, but adjusted after the computations are done in parallel!
//
// This is identical to what's done in the OpenMP version of this code, but information is shared via message passing rather
// than shared memory.
//
// Note: this program will be helpful for the Ex #4 train problem where the anti-derivatives are unknown and all functions must
//       be integrated numerically.
//
// Sam Siewert, 2/24/2023, Cal State Chico - https://sites.google.com/mail.csuchico.edu/sbsiewert/home
//
// Please use as you see fit, but please do cite and retain a link to this original source
// here or my github (https://github.com/sbsiewertcsu/numeric-parallel-starter-code)
//

#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>

#define DEBUG_TRACE

// Function tables used in Ex #3 and #4 as well as test profiles for sine and a constant
// All funcitons have 1801 entries for time = 0.0 to 1800.0
#include "timeinterp.h"
//#include "ex3.h"
//#include "sine.h"
//#include "const.h"

const int MAX_STRING = 512;
double func_to_integrate(double x);
double func_velocity(double time, double *velocity_profile, double step_size);
double Simpson(double left_endpt, double right_endpt, int num_intervals, double base_len, double *values);

int main(void)
{

    double a = 0.0;
    double b = 1800;
    int n = (b - a) * 10000;
    double step_size = 1.0 / 10000.0;
    char greeting[MAX_STRING];
    char hostname[MAX_STRING];
    char nodename[MAX_STRING];
    //char nodename[MPI_MAX_PROCESSOR_NAME];
    int comm_sz;
    int my_rank, namelen;
    // Parallel summing example
    double local_final_velocity=0.0, final_velocity=0.0;

    double *local_velocity = (double *)malloc(n * sizeof(double));
    if (local_velocity == NULL) {
        fprintf(stderr, "Failed to allocate memory for local_velocity\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double *local_displacement = (double *)malloc(n * sizeof(double));
    if (local_displacement == NULL) {
        fprintf(stderr, "Failed to allocate memory for local_displacement\n");
        free(local_velocity); // Free previously allocated memory
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int subrange, residual;
    printf("n = %d\n", n);
    // Fill in local_velocity array used to simulate a new table of values, such as
    // a velocity table derived by integrating acceleration
    //
    for(int idx = 0; idx < n; idx++)
    {
        local_velocity[idx]=0.0;
        local_displacement[idx]=0.0;
    }


    printf("Will divide up work for input table of size = %d\n", n);
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    subrange = n / comm_sz;
    residual = n % comm_sz;

    printf("Went parallel: rank %d of %d doing work %d with residual %d\n", my_rank, comm_sz, subrange, residual);

    // START PARALLEL PHASE 1: Sum original DefaultProfile LUT by rank
    //
    if(my_rank != 0)
    {
        gethostname(hostname, MAX_STRING);
        MPI_Get_processor_name(nodename, &namelen);

        // Now sum up the values in the LUT function
        // for(int idx = my_rank*subrange; idx < (my_rank*subrange)+subrange; idx++)
        // {
        //     local_final_velocity += DefaultProfile[idx];
        //     local_velocity[idx] = local_final_velocity; // Each rank has it's own subset of the data
        // }
        local_final_velocity = Simpson(my_rank*subrange, (my_rank*subrange)+subrange, subrange, step_size, &local_velocity[my_rank*subrange]);

        sprintf(greeting, "Sum of DefaultProfile for rank %d of %d on %s is %lf", my_rank, comm_sz, nodename, local_final_velocity);
        MPI_Send(greeting, strlen(greeting)+1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    else
    {

        gethostname(hostname, MAX_STRING);
        MPI_Get_processor_name(nodename, &namelen);

        // Now sum up the values in the LUT function
        // for(int idx = 0; idx < subrange; idx++)
        // {
        //     local_final_velocity += DefaultProfile[idx];
        //     local_velocity[idx] = local_final_velocity; // Each rank has it's own subset of the data
        // }
        local_final_velocity = Simpson(my_rank*subrange, (my_rank*subrange)+subrange, subrange, step_size, &local_velocity[my_rank*subrange]);
        printf("Sum of DefaultProfile for rank %d of %d on %s is %lf\n", my_rank, comm_sz, nodename, local_final_velocity);

        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(greeting, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("%s\n", greeting);
        }
    }

    // This should be the summation of DefaultProfile, which should match the spreadsheet for a train profile for dt=1.0
    MPI_Reduce(&local_final_velocity, &final_velocity, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(my_rank == 0) printf("\nRank 0 final_velocity = %lf\n", final_velocity);

    MPI_Barrier(MPI_COMM_WORLD);

    // DISTRIBUTE RESULTS: Now to correct and overwrite all local_velocity, update all ranks with full table by sending
    // portion of table from each rank > 0 to rank=0, to fill in missing default data
    if(my_rank != 0)
    {
        MPI_Send(&local_velocity[my_rank*subrange], subrange, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(&local_velocity[q*subrange], subrange, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Adjust so initial condition is ending conditon of prior sum - SUPER IMPORTANT to adjust initial condition offset
            for(int idx = q*subrange; idx < (q*subrange)+subrange; idx++)
                local_velocity[idx] += local_velocity[((q-1)*subrange)+subrange-1];
        }
               

    }
    // Make sure all ranks have the full new default table
    MPI_Bcast(&local_velocity[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //
    // END PARALLEL PHASE 1: Every rank has the same updated local_velocity table now


    // TRACE: Just a double check on the FIRST MPI_Reduce and trace output of first phase
    if(my_rank == 0)
    {
#ifdef DEBUG_TRACE
        // Now rank zero has the data from each of the other ranks in one new table
        printf("\nTRACE: Rank %d sum of local_velocity = %lf\n", my_rank, final_velocity);
            // for(int idx = 0; idx < n; idx+=100)
        //     printf("t=%f: a=%lf for v=%lf\n", idx * step_size, func_to_integrate(idx * step_size), local_velocity[idx-1] * 3.6);
#endif
    }


    // START PARALLEL PHASE 2: Now that all ranks have the new local_velocity table, we can proceed to sum all of those sums as before
    //
    // Do the next round of summing from the new table
    //
    local_final_velocity=0; //  for second integration

    if(my_rank != 0)
    {
        // Now sum up the values in the new LUT function local_velocity
        // for(int idx = my_rank*subrange; idx < (my_rank*subrange)+subrange; idx++)
        // {
        //     local_final_velocity += local_velocity[idx];
        //     local_velocity_of_sums[idx] = local_final_velocity; // Each rank has it's own subset of the data
        // }
        local_final_velocity = Simpson(my_rank*subrange, (my_rank*subrange)+subrange, subrange, step_size, local_velocity);
    }
    else
    {
        // Now sum up the values in the new LUT function local_velocity
        // for(int idx = 0; idx < subrange; idx++)
        // {
        //     local_final_velocity += local_velocity[idx];
        //     local_velocity_of_sums[idx] = local_final_velocity; // Each rank has it's own subset of the data
        // }
        local_final_velocity = Simpson(my_rank*subrange, (my_rank*subrange)+subrange, subrange, step_size, local_velocity);
    }

    // This should be the summation of the sums, which should match the spreadsheet for a train profile for dt=1.0
    MPI_Reduce(&local_final_velocity, &final_velocity, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if(my_rank == 0) {
        printf("\nFinal Velocity = %lf\n", final_velocity * 3.6);
    }

    // DISTRIBUTE: Finally correct and overwrite all local_velocity_of_sums, update all ranks with full table by sending
    // portion of table from each rank > 0 to rank=0, to fill in missing default data
    if(my_rank != 0)
    {
        MPI_Send(&local_displacement[my_rank*subrange], subrange, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else
    {
        for(int q=1; q < comm_sz; q++)
        {
            MPI_Recv(&local_displacement[q*subrange], subrange, MPI_DOUBLE, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Adjust so initial condition is ending conditon of prior sum - SUPER IMPORTANT to adjust initial condition offset
            for(int idx = q*subrange; idx < (q*subrange)+subrange; idx++)
                local_displacement[idx] += local_displacement[((q-1)*subrange)+subrange-1];
        }
        printf("Final displacement = %f\n", local_displacement[n-1] * 3.6);
        printf("Midpoint displacement = %f\n", local_displacement[n/2] * 3.6);
        printf("quarter displacement = %f\n", local_displacement[n/4] * 3.6);
    }
    // Make sure all ranks have the full new default table
    // MPI_Bcast(&local_velocity[0], tablelen, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //
    // END PHASE 2: Every rank has the same updated local_displacement table now


    // TRACE: Final double check on the SECOND MPI_Reduce and trace output of second phase
//     if(my_rank == 0)
//     {
// #ifdef DEBUG_TRACE
//         // Now rank zero has the data from each of the other ranks in one new table
//         printf("\nTRACE: Rank %d sum of local_velocity = %lf\n", my_rank, final_velocity);
//         for(int idx = 0; idx < tablelen; idx+=100)
//             printf("t=%d: a=%lf for v=%lf and p=%lf\n", idx, DefaultProfile[idx-1], local_velocity[idx-1], local_displacement[idx-1]);
// #endif
//     }

    MPI_Finalize();
    return 0;
}


// Function to perform Simpson's rule integration and populate an array if needed
double Simpson(double left_endpt, double right_endpt, int num_intervals, double base_len, double *values)
{
    double estimate, x;
    int i;

    if (num_intervals % 2 != 0)
    {
        printf("Simpson's rule requires an even number of intervals.\n");
        return 0.0;
    }
    double left_time = left_endpt * base_len;
    double right_time = right_endpt * base_len;

    estimate = func_to_integrate(left_time) + func_to_integrate(right_time);

    // Sum up contributions from the internal points
    for (i = 1; i <= num_intervals - 1; i++)
    {
        x = left_time + i * base_len;
        if (i % 2 == 0)
        {
            estimate += 2 * func_to_integrate(x);
        }
        else
        {
            estimate += 4 * func_to_integrate(x);
        }
        values[i] = estimate * base_len / 3.0;
    }

    estimate = estimate * base_len / 3.0;
    return estimate;
}

double Simpson_velo(double left_endpt, double right_endpt, int num_intervals, double base_len, double *values)
{
    double estimate, x;
    int i;

    if (num_intervals % 2 != 0)
    {
        printf("Simpson's rule requires an even number of intervals.\n");
        return 0.0;
    }
    double left_time = left_endpt * base_len;
    double right_time = right_endpt * base_len;

    estimate = func_to_integrate(left_time) + func_to_integrate(right_time);

    // Sum up contributions from the internal points
    for (i = 1; i <= num_intervals - 1; i++)
    {
        x = left_time + i * base_len;
        if (i % 2 == 0)
        {
            estimate += 2 * func_to_integrate(x);
        }
        else
        {
            estimate += 4 * func_to_integrate(x);
        }
        values[i] = estimate * base_len / 3.0;
    }

    estimate = estimate * base_len / 3.0;
    return estimate;
}

double func_to_integrate(double x)
{
    return faccel(x);
}

double func_velocity(double time, double *velocity_profile, double step_size)
{
    // return fvelocity(time, velocity_profile, step_size);
    return 0;
}