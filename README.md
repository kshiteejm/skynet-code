# skynet-code

## Problem
1. Description: 
* Network Topology = Graph = Adjacency Matrix over all nodes in the network. 
* Flows = (src, dst) = Partial Routes = Dynamic State
* Policies =  
Reachability to dst. Matrix of size #flows * #nodes. 1 for (flow_i, src_i) and (flow_i, dst_i).    
(Soft) Isolation Policy = Flow i and j should not share any link. Matrix of size #flows * #flows. (flow_i, flow_j) = 1 and (flow_j, flow_i) = 1.  
2. Decompose this problem:
* Initial State (s_0): Each flow is at node src. Matrix of size #flows * #nodes. (flow_i, src_i) = 1 
* We extend each flow by one hop at a time.
* Round Robin over Flows
* State: Topology, Partial Routes for each Flow, Current Active Flow, Next Hops (set of neighboring nodes which have not been visited and are reachable to the destination) 
* Action: Which Next Hop to Choose? 

### Brain
1. train_queue = interactions with the environment = current state (s), action (a), immediate reward (r), next state (s’)
2. Reward: discounted over n-steps with gamma discounting factor = sum from i = 0 to n-steps gamma^i * r_i
3. Graph Neural Net (Policy Model i.e. Actor-Critic Model) = inputs = state and output = next hop

### Environment
1. env.reset = state_0 = Topology is always constant.  
The flows and policies have an upper bound =  
You pick the same set of flows and same set of policies, OR  
You randomly pick set of flows and set of policies.  
* Agent = Wrapper Around the Brain Model
2. action = agent.act(current_state) ← does inference on the policy i.e., actor model
3. next_state, reward = env.step(action)
4. repeated to get an episode (populate the train_queue)

### Optimizer
1. Batched Training : When train_queue is sufficiently full
2. brain.optimize(): backprop on the actor-critic model → you get updated actor-critic model


### Main
1. while True:  
Optimizer+Brain initialization => start filling the train_queue  
Environment => run episodes with the current version of the brain i.e., the actor-critic policy model  
After recording episodes and filling the train_queue you train and update Brain 
