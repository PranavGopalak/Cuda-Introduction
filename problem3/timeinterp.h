#ifndef TIMEINTERP_H
#define TIMEINTERP_H

// table look-up for acceleration profile given and velocity profile determined
//
// Note: for 2 functions (2 trains) we would want to make 2 different versions of this
//       function or better yet, pass in the table to use.
//
double table_accel(int timeidx);


// indirect generation of acceleration or velocity at any time with table interpolation
//
// Note: for 2 functions (2 trains) we would want to make 2 different versions of this
//       function that each uses the correct table.
double faccel(double time);

#endif // TIMEINTERP_H