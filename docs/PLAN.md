# Project Planning


### Backlog
- [ ] Look at how SWE-Bench works exactly, should we train for that behavior end2end?
- [ ] Update DATASETS.md with everything
  - From overleaf
  - From emails
  - From docs (both mine and minutes)
- [ ] Make hf cache (and uv if simple) point to my project directory instead of my home directory (on slurm)
  - Relatively easy to just do export TRANSFORMERS_CACHE=/proj/berzileus-2024-336/x_bjabj/huggingface
  - And do that both in my .zshrc and the apptainer environment
- [ ] Reply here, community discussion on verifiers
  - https://github.com/willccbb/verifiers/discussions/35
- [ ] Info dump on the nano-R1 discussion
  - https://github.com/nano-R1/resources/discussions/1


## March 17 - March 23, 2025
Was a bit sick during the weekened which is when I am normally the most productive

#### Tasks
- [x] Fix apptainer and test_job.sh
  - I know it works to cp ttc.sif to my project directory then do apptainer exec ttc.sif uv run ...
  - Maybe some discrepancy between that and what test_job.sh is doing
  - Or some complexity I'm not aware about in srun
- [x] Fix some refactor mismatches
- [ ] Get train_job.sh working
  - Don't really have a task worth training on right now but it would be very nice to cross this off the list
- [x] smol-stack-docstrings
  - now we can train a diff matching implementation generating model 
- [ ] Train on SWE-Bench-lite on diff matching
  - Find out how they normally put the codebase into the context window
  - Otherwise we could just run repomix
- [ ] Read LLM Post-Training: A Deep Dive into Reasoning Large Language Models: https://arxiv.org/pdf/2502.21321
- [ ] Read MUFIN paper
- [ ] Read S∗: Test Time Scaling for Code Generation: https://arxiv.org/pdf/2502.14382
- [ ] Setup Defect4J as reward signal and train on it

#### Ideas
- What if we use AST to extract docstrings of python code on the Stack.
  - Sufficiently detailed docstrings can be mutated (via LLM) into a detailed description of the desired functionality of a function or class
  - We then remove said function/class but provide the rest of the code as context


## March 10 - March 16, 2025

#### Supervisor Updates
- Using test cases as rewards may require alot of SWE to run each test case on Kubernetes
  - Issues with scalability

#### Tasks
- [x] Fix diff.py, it is VERY sloppy
- [x] Fix reward.py, it sloppy
  - Get rid of get_diff entirely
  - Make correctness and quality rewards as separate as possible
  - Merge strict quality with partial quality, since the latter smoothly transitions from 0-1
  - Also add a method to diff.py to compare two diffs of the same type 
- [x] Side quest: Fast QwQ-32B inference on 1 of KTH's H100
- [x] Check out verifiers, seems like exactly what I need, could even contribute to the repo
  - paralell execution env where we can call dockerized tests?
- [x] Check out lighteval, would be cool to check e.g. humaneval before and after
  - looks good, much better than deepeval which I was using before


#### Thoughts
- Should we do have an agent scaffolding like SWE-Agent / OpenHands / Agentless
- Would be really cool to reinforce tool usage but that requires a lot of engineering
- Bug fix diff matching is probably almost equivalent to running the tests
  - At least for simple bugs which only require changing a couple lines
- Perhaps a more interesting task is implementing something new with well defined behaviour
  - Then oracle patch diff matching is a vastly different objective from passing the tests
  - Can we create a synthetic dataset for this?


## March 3 - March 9, 2025

#### Supervisor Updates
- Notes from meeting:
  - I need to scope out my project more, decide what task I am optimizing for exactly
  - Is that code execution RL? 
  - Will I be doing a ML paper? Data curation and cleaning paper / agentic coder paper?
  - Am I targeting SWE-Bench? RepairBench?

#### Tasks
- [x] Scope out my project
- [ ] Look at how SWE-Bench works exactly, should we train for that behavior end2end?
- [ ] Fix SLURM scripts (get a basic training run going)

#### Thoughts
- Should I keep on using GRPOTrainer?
- Should I fork verifiable envs or create my own (execution consideration)
