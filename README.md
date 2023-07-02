# The schedule of implementation

All newly written functions will be uploaded to the "upload folder" after completing each phase. We could use them to replace old functions in SDPR to gradually make it SDPRX. Once I have completed one, I will check the checkbox to show my progress

## Phase 0 Set up the environment

- [x] Make sure that the complier works
- [x] Make sure that default packages (such as GSL, fstream, sstream etc.) works on you computer


This phase completes upon the condition: Pass the tests assigned for default packages with satisfactory output
## Phase 1 Read in the data
- [x] Know what files are read in
- [x] Figure out which of the parsing functions needs modification
- [x] Run the original code to read in the one population data
- [x] Modify the code to read in the data that needs to be read in with SDPRX


This phase completes upon the condition: All test data of SDPRX can be successfully read in with the modified function, based on satisfying output (such as head)
## Phase 2 Data preprocessing
- [x] Know how the read-in data are processed as variables (in stack or heap? Any pointers needed?)
- [x] Write steps to preprocess the readin data


This phase completes upon the condition: After preprocessing the test data, new C++ code give the same output as the Python code
## Phase 3 Set up the state object and its initialization
- [x]  Sketch out the variables needed in the class and their types or dimensions
- [x]  Build the class
- [x]  Design the initiation function
- [x]  Complete initiation

## Phase 3.5 Implementation of LD matrix evaluation
- [ ] Read the code in SDPRX
- [ ] Test the code for SDPRX on HPC
- [ ] Read the code in SDPR
- [ ] Modify the code for SDPR
- [ ] Test the code for SDPR on small scale dataset
- [ ] Test the code for SDPR on large scale dataset

This phase completes upon the condition: After initiation, new C++ code give the same output as the Python code
## Phase 4 Set up the sampling functions
- [ ]  assignment
- [ ]  beta
- [ ]  eta
- [ ]  prob for each population
- [ ]  alpha
- [ ]  stick-breaking parts
- [ ]  sufficient stats
- [ ]  The stick-breaking one
- [ ]  Integration of the sampling functions to complete one MCMC cycle


This phase completes upon the condition: With testing, each function give the same output as the Python code

## Phase 5 Integration
- [ ]  Design the output of the code
- [ ]  Integrate all implemented function such as input and threads
- [ ]  Write the makefile
- [ ]  Code documentation and commenting
- [ ]  Test the program with multiple applications

