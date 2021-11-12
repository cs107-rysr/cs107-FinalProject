# cs107-FinalProject Milestone 2b

1. Tasks each group member has been assigned to for Milestone2

Raymond J: Configure TravisCI to run tests on basic operations on the spladtool_forward.tensor class.  Fix any bugs on the dunder methods implemented in the tensor class and make sure the forward mode calculations and gradients adhere to the input data type dimensionality.  

Yuanbiao Wang: Based on the feedback of running unittests, update the structure and basic operations of forward mode automatic differentiation if needed. Propose for additional features, like optimizer and GUI.

Rye Julson: Implement elementary functions, for example, exponentials and logarithms, and square root for both scalar inputs and vector inputs and make sure the implementation passed all unittests. Extend the documentation of Milestone 1 regarding new update.

Shihan Lin: Implement all comparison operators (<, <=, >, >=, ==, !=) for both scalar inputs and vector inputs and make sure the implementation passed all unittests. Implememt elementary functions together with Rye.


2. What has each group member done since Milestone 1

Raymond J: Wrote tests for all basic operations.  Added a comparison operator to compare tensor class objects in the unittests.  Configured the travis.yml file to run tests in tests module.  Restructured spladtool_forward into /lib and /tests module.  

Yuanbiao Wang: Constructed the whole structure of forward mode automatic differentiation. Implemented the basic operations (addition, subtraction, multiplication, division, power and negation). Construct the basic outline of reverse mode automatic differentiation.

Rye Julson: Updated the document of Milestone 1 with a feedback section to reflect the feedback given by TF. 

Shihan Lin: Implemented part of the comparison operators (<, >, ==) for scalar inputs, which passed the basic assertion tests.
