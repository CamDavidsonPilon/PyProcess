PyProcess
==========

PyProcess is a Python class library used to exactly simulate stochastic processes, and their properties.

Using this library, you can simulate the following random processes:

Continuous Diffusions
- Brownian Motion
- Geometric Brownian Motion
- CEV
- CIR
- Square Bessel Process 
- Ornstein Uhlenbeck process
- Time-integrated Ornstein Uhlenbeck process
- Levy Processes
- Bessel Process (coming soon)
- Fractional Brownian Motion (coming soon)

Jump Diffusions
- Gamma process
- Variance-gamma process
- Geometric Gamma process
- Inverse Gaussian process *NEW*
- Normal Inverse Gaussian process *NEW*

Step Processes
- Renewal process
- Poisson process
- Compound poisson process
- marked-poisson process
- Fractional poisson process (coming soon)


*See fun examples of the processes you can simulate [here] (http://pyprocess.70percentfatfree.com)*



How PyProcess Works
--------------------

To create a stochastic project instance, you need to give the class the specific parameters and any time or space constraints.
For example, if we wish to create a non-standard Brownian motion instance (named Wiener_motion) which has parameters 
'mu' and 'sigma', we use:


    parameters = {'mu':-1, 'sigma':2}
    time_space_constraints = {'startTime': 0, 'startPosition':2}
    process = Wiener_process( parameters, time_space_constraints )

    process.get_mean_at(10)
    # -8.0

    process.generate_position_at(10)
    # -1.96953969194

    times = range(10)
    process.generate_sample_path(times)
    # [(0, 2.0), (1, 3.5504434920777763), (2, 2.8862861385211374), (3, 2.835569957404396), (4, 3.114651632088956), (5, 2.983357771156665), (6, 2.55760572607476), (7, 2.2188866076280434), (8, 6.528248392922263), (9, 8.349381153333603)]


Most importantly about this library is the ability to condition process on endpoints too. For example, suppose we want to 
tie the browian motion above at the endpoint (10,0) (ie. at time 10, the process is at 0). This is simple, continuing from 
above we add the constraints to time_space_constraints:

    time_space_constraints['endTime']=10
    time_space_contraints['endPosition']=10
    process = Wiener_process( parameters, time_space_constraints )

    process.get_mean_at(9)
    # 0.2

    process.generate_position_at(9)
    # -1.33602629698


    times = range(10)
    process.generate_sample_path(times)
    #[(0, 2.0), (1, 2.031211212004885), (2, 0.5046601471443315), (3, -0.39302356242482717), (4, -1.4515387073784005), (5, 0.023279992315228704), (6, 0.5196539154794165), (7, -1.7891276719165035), (8, -4.564491110511355), (9, -3.2776199543473057), (10, 0.0)]


There are more examples and tricks in the examples.py file.


Lastly
-----------------
email me at cam.davidson.pilon@gmail.com and visit me at [camdp.com](http://www.camdp.com) and follow me at [cmrn_dp](http://twitter/cmrn_dp)
            
        