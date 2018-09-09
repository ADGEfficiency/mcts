
## `multiprocessing` Python docs - (https://docs.python.org/dev/library/multiprocessing.html)

## An introduction to parallel programming using Python's multiprocessing module
- https://sebastianraschka.com/Articles/2014_multiprocessing.html

Two common approaches

- via threads
- via multiple processes

Submitting jobs to different threads - this can be thought of as sub-tasks of a single process.  Will usually have access to same memory - can lead to conflicts if improper synchronization (ie writing to same memory at the same time)

Safer approach = submit multiple processes with separate memory locations (distributed memory).  Each process is totally independent.

Python's `multiprocessing` module can submit multiple processes
