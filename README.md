# The schedule of implementation

All newly written functions will be uploaded to the "upload folder" after completing each phase. We could use them to replace old functions in SDPR to gradually make it SDPRX. Once I have completed one, I will check the checkbox to show my progress

## Phase 0 Set up the environment

- [x] Make sure that the complier works
- [x] Make sure that default packages (such as GSL, fstream, sstream etc.) works on you computer
This phase completes upon the condition: Pass the tests assigned for default packages with satisfactory output
## Phase 1 Read in the data
- [ ] Figure out which of the parsing functions needs modification
- [ ] Run the original code to read in the one population data
- [ ] Modify the code to read in the data that needs to be read in with SDPRX
This phase completes upon the condition: All test data of SDPRX can be successfully read in with the modified function, based on satisfying output (such as head)
## Phase 2 Data preprocessing
- [ ] Know how the read-in data are processed as variables (in stack or heap? Any pointers needed?)
- [ ] Write steps to preprocess the readin data
This phase completes upon the condition: After preprocessing the test data, new C++ code give the same output as the Python code
## Phase 3 Set up the state object and its initialization
- [ ]  Sketch out the variables needed in the class and their types or dimensions
- [ ]  Build the class
- [ ]  Design the initiation function
This phase completes upon the condition: After initiation, new C++ code give the same output as the Python code
## Phase 4 Set up the sampling functions
- [ ]  Design the samling functions one by one
- [ ]  assignment
- [ ]  beta
- [ ]  eta
- [ ]  prob for each population
- [ ]  alpha
- [ ]  stick-breaking parts
- [ ]  The stick-breaking one
This phase completes upon the condition: With testing, each function give the same output as the Python code
