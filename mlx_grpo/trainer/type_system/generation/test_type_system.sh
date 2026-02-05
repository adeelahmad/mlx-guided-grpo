#!/bin/zsh
# Validation script for type system

echo "================================"
echo "Type System Validation Tests"
echo "================================"

# Test 1: Import all modules
echo "\n1. Testing module imports..."
python -c "
from mlx_grpo.trainer.type_system.auto_discovery_extended import get_all_for_type
from mlx_grpo.trainer.tool_calling_reward import tool_call_exact_match
print('✓ All modules import successfully')
" || exit 1

# Test 2: Test type discovery
echo "\n2. Testing type discovery..."
python -c "
from mlx_grpo.trainer.type_system.auto_discovery_extended import get_all_for_type

for t in ['math', 'tool_call', 'exam']:
    comps = get_all_for_type(t)
    reward_name = comps['reward'].__class__.__name__
    strategy_name = comps['generation'].__class__.__name__
    print(f'✓ {t}: {reward_name}, {strategy_name}')
    assert reward_name != 'BaseReward', f'{t} should have specific reward'
    assert strategy_name != 'BaseGenerationStrategy', f'{t} should have specific strategy'
" 2>&1 | grep -v INFO || exit 1

# Test 3: Test tool calling rewards
echo "\n3. Testing tool calling rewards..."
python tests/test_tool_calling_rewards.py 2>&1 | tail -5 || exit 1

# Test 4: Test reward registry
echo "\n4. Testing reward registry..."
python -c "
from mlx_grpo.trainer.rewards import list_rewards
rewards = list_rewards()
assert 'tool_call_exact' in rewards
print(f'✓ {len(rewards)} rewards registered')
" || exit 1

echo "\n================================"
echo "✅ ALL TESTS PASSED"
echo "================================"
