import typer
import numpy as np
from omegaconf import OmegaConf

from rlrs.datasets.food import FoodSimple
from rlrs.envs.food_offline_env import FoodOrderEnv
from rlrs.recommender import Recommender

AYAMPP_LITE_CFG = "configs/ayampp_small_base_infer.yaml"


def train_test_split(dataset,cfg): 
    user_num = dataset.num_users()
    test_ratio = cfg["test_ratio"]
    eval_user_num = int(user_num * test_ratio)
    eval_user_list = dataset.users_[eval_user_num:]
    return eval_user_list


def evaluate(recommender, env ,top_k, verbal = False): 
    steps = 0
    mean_precision = 0
    user_id, items_ids, done, episode_reward = env.reset()

    if verbal:
        print(f'user_id : {user_id}, user_history_length:{len(env.db.get_user_history_length(user_id))}')
        '''TODO: missing get_items_names equivalent'''
        print('history items : \n', np.array(env.get_items_names(items_ids)))
    
    if not done:
        recommended_item = recommender.recommend()
        if verbal:
            print(f'recommended items ids : {recommended_item}')
            '''TODO: missing get_items_names equivalent'''
            print(f'recommened items : \n {np.array(env.get_items_names(recommended_item), dtype=object)}')
        # Calculate reward & observe new state (in env)
        '''our current step has no concept of TOPK'''
        _, next_items_ids, done , reward = env.step(recommended_item, top_k=top_k)
        '''TODO: need the list of rewards in different items recommended for eval'''
        correct_list = [1 if r > 0 else 0 for r in reward]
        
        #precision
        correct_num = top_k - correct_list.count(0)
        mean_precision += correct_num/top_k
            
        reward = np.sum(reward)
        items_ids = next_items_ids
        episode_reward += reward
        steps += 1
        
        if verbal:
            print(f'precision : {correct_num/top_k}, reward : {reward}')
            print()
    
    if verbal:
        print(f'precision : {mean_precision/steps}, episode_reward : {episode_reward}')
        print()
    
    return mean_precision/steps


def main(): 
    cfg = OmegaConf.load(AYAMPP_LITE_CFG)
    dataset = FoodSimple.from_folder(cfg["input_data"])
    recommender = Recommender(env, cfg)
    TOP_K = cfg["topk"]
    SHOW_EVAL = cfg["show_evaluation"]
    sum_precision = 0

    eval_user_list = train_test_split(dataset,cfg)


    for i, user_id in enumerate(eval_user_list):
        '''TODO: update env for real'''
        env = FoodOrderEnv(
            dataset, state_size=cfg["state_size"], 
            user_id=user_id, done_count = 6
            )
        '''TODO: check if the below is needed for our recommender'''
        '''BELOW is to init the recommender to generate eval'''
        recommender = DRRAgent(env, users_num, items_num, STATE_SIZE)
        recommender.actor.build_networks()
        recommender.critic.build_networks()
        recommender.load_model(saved_actor, saved_critic)

        '''verbal information for show and tell (presentation) & need to dump somewhere'''
        precision = evaluate(recommender, env, verbal=True, top_k=TOP_K) 
        sum_precision += precision
        
        if SHOW_EVAL > 0:
            if i > SHOW_EVAL:
                break

if __name__ == "__main__":
    typer.run(main)
