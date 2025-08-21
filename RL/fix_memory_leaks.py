#!/usr/bin/env python3
"""
Fix memory leaks caused by growing computation graphs in RL code.

Key issues found:
1. Hidden states stored without detaching (keeps entire graph)
2. Losses appended without .item() (keeps graph)
3. States in replay buffers not detached
"""

import os
import sys

def fix_hidden_state_storage():
    """Fix hidden state storage to detach from computation graph"""
    
    fixes = [
        # DQN Agent - detach hidden state when storing
        {
            'file': 'model/dqn_agent.py',
            'old': '                    q_values, self.hidden_state = self.q_network.forward_single_step(\n                        state_tensor, self.hidden_state\n                    )',
            'new': '                    q_values, new_hidden = self.q_network.forward_single_step(\n                        state_tensor, self.hidden_state\n                    )\n                    # Detach hidden state to prevent graph accumulation\n                    if new_hidden is not None:\n                        self.hidden_state = tuple(h.detach() for h in new_hidden)\n                    else:\n                        self.hidden_state = new_hidden'
        },
        
        # NGU Agent - detach hidden state
        {
            'file': 'model/ngu_agent.py',
            'old': "            # Update hidden state\n            self.hidden_state = result['hidden_state']",
            'new': "            # Update hidden state (detached to prevent graph growth)\n            if result['hidden_state'] is not None:\n                self.hidden_state = tuple(h.detach() for h in result['hidden_state'])\n            else:\n                self.hidden_state = result['hidden_state']"
        },
        
        # R2D2+Agent57 - detach hidden states
        {
            'file': 'model/r2d2_agent57_hybrid.py',
            'old': "            # Update hidden state\n            self.hidden_states[policy_id] = result['hidden_state']",
            'new': "            # Update hidden state (detached to prevent graph growth)\n            if result['hidden_state'] is not None:\n                self.hidden_states[policy_id] = tuple(h.detach() for h in result['hidden_state'])\n            else:\n                self.hidden_states[policy_id] = result['hidden_state']"
        }
    ]
    
    return fixes

def fix_loss_storage():
    """Fix loss storage to use .item() instead of tensor"""
    
    fixes = [
        # R2D2+Agent57 - use .item() for loss
        {
            'file': 'model/r2d2_agent57_hybrid.py',
            'old': "            self.training_stats['losses'].append(loss)",
            'new': "            self.training_stats['losses'].append(loss.item() if hasattr(loss, 'item') else loss)"
        },
        
        # Distributed DQN - use .item() for losses
        {
            'file': 'model/distributed_dqn_agent.py',
            'old': "                            self.training_stats['losses'].append(loss)",
            'new': "                            self.training_stats['losses'].append(loss.item() if hasattr(loss, 'item') else loss)"
        },
        {
            'file': 'model/distributed_dqn_agent.py',
            'old': "                        self.training_stats['losses'].append(loss)",
            'new': "                        self.training_stats['losses'].append(loss.item() if hasattr(loss, 'item') else loss)"
        },
        
        # NGU Agent - already using .item() for total_loss, but fix episode_losses
        {
            'file': 'model/ngu_agent.py',
            'old': "                    episode_losses.append(loss)",
            'new': "                    episode_losses.append(loss.item() if hasattr(loss, 'item') else loss)"
        },
        
        # DQN Agent - use .item() for losses
        {
            'file': 'model/dqn_agent.py',
            'old': "                        episode_losses.append(loss)",
            'new': "                        episode_losses.append(loss.item() if hasattr(loss, 'item') else loss)"
        }
    ]
    
    return fixes

def fix_replay_buffer_storage():
    """Ensure states are detached when stored in replay buffers"""
    
    fixes = [
        # Data buffer - detach states before storing
        {
            'file': 'model/data_buffer.py',
            'old': "    def push(self, state, action, reward, next_state, done):\n        self.buffer.append((state, action, reward, next_state, done))",
            'new': "    def push(self, state, action, reward, next_state, done):\n        # Detach tensors if they have gradients to prevent graph accumulation\n        if hasattr(state, 'detach'):\n            state = state.detach()\n        if hasattr(next_state, 'detach'):\n            next_state = next_state.detach()\n        self.buffer.append((state, action, reward, next_state, done))"
        }
    ]
    
    return fixes

def main():
    """Generate all fixes"""
    
    print("Memory Leak Fixes for RL Code")
    print("=" * 60)
    
    all_fixes = []
    
    # Collect all fixes
    all_fixes.extend(fix_hidden_state_storage())
    all_fixes.extend(fix_loss_storage())
    all_fixes.extend(fix_replay_buffer_storage())
    
    print(f"\nFound {len(all_fixes)} critical memory leak issues to fix:")
    print()
    
    for i, fix in enumerate(all_fixes, 1):
        print(f"{i}. {fix['file']}")
        print(f"   Issue: Storing tensors without detaching/converting to scalar")
        print()
    
    print("\nThese fixes will:")
    print("1. Detach hidden states when storing to prevent graph accumulation")
    print("2. Convert losses to scalars with .item() before storing")
    print("3. Detach states in replay buffers to prevent gradient storage")
    print()
    print("This should significantly reduce memory usage and prevent OOM errors.")
    
    # Save fixes to file for review
    import json
    with open('memory_leak_fixes.json', 'w') as f:
        json.dump(all_fixes, f, indent=2)
    print("\nFixes saved to memory_leak_fixes.json for review")

if __name__ == '__main__':
    main()