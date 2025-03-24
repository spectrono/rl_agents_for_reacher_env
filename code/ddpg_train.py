import torch
import numpy as np
import matplotlib.pyplot as plt

from environment import setup_reacher_environment, UnityMLEnvironemtAdapter
from collections import deque
from time import time
from datetime import timedelta
from ddpg_agent import DDPGAgent
from pathlib import Path
from os.path import join


def plot_scores(scores, result_directory_path, nb_of_episodes_for_pure_data_collection, episode_count=0, title='DDPG Training', show_plot=False):
    """
    Plot scores from training.
    
    Args:
        scores (list): List of scores
        result_directory_path(Path): Where to save the figure
        nb_of_episodes_for_pure_data_collection (int): Draws a vertical line to show up to which timestep only data was collected and nor learning happend
        episode_count(int): Number of episoded
        title (str): Title for the plot
        show_plot (bool): If false, the figure is saved to disk, only
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot raw scores
    ax.plot(np.arange(len(scores)), scores, alpha=0.3, label='Scores')
    
    # Plot rolling mean
    rolling_mean = np.convolve(scores, np.ones(100)/100, mode='valid')
    ax.plot(np.arange(len(rolling_mean)) + 99, rolling_mean, label='Moving Average (100 episodes)')
    
    # Add labels and legend
    ax.set_xlabel('Episode (n)')
    ax.set_ylabel('Score')
    if episode_count == 0:
        ax.set_title(title)
    else:
        ax.set_title(title + " episode idx: " + str(episode_count))

    ax.legend()

    # Mark episode for pure data collection
    plt.axvline(x=nb_of_episodes_for_pure_data_collection, color='g', linestyle='--', label='Vertical Line')

    # Annotate the vertical line
    max_y = np.max(scores)
    annotation_position_y = 0.95 * max_y
    plt.text(nb_of_episodes_for_pure_data_collection, annotation_position_y, f'Pure data collection episodes  (n<={nb_of_episodes_for_pure_data_collection})', rotation=90, verticalalignment='top', horizontalalignment='left', label="Pure data collection episodes")
    
    plt.savefig(join(result_directory_path, f"{title.replace(' ', '_')}_episode_idx_{episode_count:06d}.png"))

    if show_plot:
        plt.show()
    
    plt.close()


def check_performance(agent, env_adapter, max_steps_count, min_score_per_episode_for_success):
    '''
    Realizes a performance test with the agent's noise model turned off.

    Useful to test if the agent is learning anything meaningful and also to estimate the influence of the noise model.

    agent(DDPGAgent): The agent to test
    env_adapter(UnityMLEnvironemtAdapter): Adapter to control the environment  
    max_steps_count: How many time steps to test the agent
    min_score_per_episode_for_success: Minimum desired test threshold for solving the environment
    '''
    # Reset environment
    state = env_adapter.reset()

    score = 0
    for _ in range(max_steps_count):

        action = agent.act(state, do_explore=False)

        exp = env_adapter.step(action)

        # Update state and score
        state = exp[0]
        score += exp[1]
        
        if exp[2]:  # done
            break

    if score > min_score_per_episode_for_success:
        return True, score
    else:
        return False, score


def train_ddpg_on_reacher_single_arm(
        seed,
        device_type,
        max_episodes_count,
        max_steps_count,
        pure_data_collection_episodes_nb,
        learn_every_x_steps,
        learning_steps,
        print_every,
        plot_every,
        save_every,
        check_performance_every,
        result_directory_path,
        score_window_size=100,
        target_score=30.0):
    
    '''
    Main call to train the DDPG agent in the single arm reacher environment.

    seed(int): initialization seed 
    device_type: cuda, cpu, ... 
    max_episodes_count(int): Maximum number of episodes to train.
    max_steps_count(int): Maximum number of steps per episode.
    pure_data_collection_episodes_nb(int): Number of episodes solely used for gathering data. No training in these episodes.
    learn_every_x_steps(int): Controlls the time steps at which a learning is executed.
    learning_steps(int): Controlls how many learning steps are done when the learning is executed.
    print_every(int): Controlls the frequency of training progress outputs on the console.
    plot_every(int): Controlls the frequency of training progress plots generation.
    save_every(int): Controlls the frequency of ddpg agent network backups to disk.
    check_performance_every(int): Controlls the frequency of the current agent's performance with the noise model deactivated.
    result_directory_path(Path): Folder to save results in.
    score_window_size(int): Size of rolling mean to rate the training progress/success.
    target_score(float): Min value of the rolling mean which has to be reached to consider the environment as solved.

    '''
    # Prevent screen blinking for unshown matplot plots which are closed on plt.close()
    plt.ioff()

    # Environment setup
    env_reacher, brain_name, agent_count, action_size, observation_size = setup_reacher_environment()
    env_adapter = UnityMLEnvironemtAdapter(env_reacher, brain_name, agent_count)

    # Agent setup
    agent = DDPGAgent(seed, device_type, observation_size, action_size, learn_every_x_steps=learn_every_x_steps, learning_steps=learning_steps)

    # Monitoring setup
    scores = []
    scores_window = deque(maxlen=score_window_size)
    best_score = 0

    print("\n>>>>>>>>>>>>>>> Starting Training <<<<<<<<<<<<<<<\n\n")
    training_time_start = time()
    for episode_idx in range(1, max_episodes_count+1):
        
        # Monitoring
        training_time_episode_start = time()
        score = 0

        # Reset environment
        state = env_adapter.reset()

        # Reset the agent (this will reset the noise model, in case one was defined!)
        agent.reset()

        for _ in range(max_steps_count):

            action = agent.act(state, do_explore=True)
            exp = env_adapter.step(action)

            if episode_idx < pure_data_collection_episodes_nb:
                agent.step(state, action, exp[1], exp[0], exp[2], do_learn=False)  # state, action, reward, state_next, done, do_learn=False
            else:
                agent.step(state, action, exp[1], exp[0], exp[2], do_learn=True)  # state, action, reward, state_next, done, do_learn=True

            # Update state and score
            state = exp[0]
            score += exp[1]
            
            if exp[2]:  # done
                break
        
        # Save score and print progress
        scores_window.append(score)
        scores.append(score)
        
        episode_elapsed_time = int(time() - training_time_episode_start)

        # Track best score
        if score > best_score:
            best_score = score
            agent.save(join(result_directory_path, 'best_agent.pth'))
        
        # Save periodically
        if episode_idx % save_every == 0:
            agent.save(join(result_directory_path,f'agent_episode_{episode_idx}.pth'))
            
        # Print progress
        if episode_idx % print_every == 0:
            training_time_current  = time()
            training_duration_current = int(training_time_current - training_time_start)
            duration_as_string = str(timedelta(seconds=training_duration_current))
            print(f'Episode {episode_idx}\tAverage Score: {np.mean(scores_window):.2f}\tBest Score: {best_score:.2f}\t time for latest episode: {episode_elapsed_time} (s)\t overall training time: {duration_as_string}')
        
        if (episode_idx % plot_every) == 0:
            plot_scores(scores, result_directory_path, pure_data_collection_episodes_nb, episode_count=episode_idx)          

        if (episode_idx % check_performance_every) == 0:
            is_performance_sufficient, perfornce_test_score = check_performance(agent, env_adapter, max_steps_count, target_score)
            print(f'Episode {episode_idx}\t> Performance check <\tScore: {perfornce_test_score:.2f}\tis successful: {is_performance_sufficient}')
            if is_performance_sufficient:
                agent.save(join(result_directory_path,f'successful_performance_test_agent_episode_{episode_idx}.pth'))

        # Check if environment is solved
        if np.mean(scores_window) >= target_score:
            print(f'\nEnvironment solved in {episode_idx} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            agent.save(join(result_directory_path, 'solved_agent.pth'))
            break            

    training_time_end = time()
    training_duration = int(training_time_end - training_time_start)
    duration_as_string = str(timedelta(seconds=training_duration))
    print(f'Training took: {duration_as_string}')

    env_reacher.close()

    return scores

def main():

    # Setup
    result_directory_path = Path('results')
    result_directory_path.mkdir(exist_ok=True)
    device_type                      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device_type}")

    # Set hyperparameter
    max_episodes_count               = 10000
    max_steps_count                  = 999
    pure_data_collection_episodes_nb =  30
    learn_every_x_steps              =   5
    learning_steps                   =  10

    # Set monitoring parameter
    print_every                      =  10
    plot_every                       =  25
    save_every                       =  30
    check_performance_every          =  15

    scores = train_ddpg_on_reacher_single_arm(
        42,           # random seed
        device_type,  # cuda, cpu
        max_episodes_count=max_episodes_count,
        max_steps_count=max_steps_count,
        pure_data_collection_episodes_nb=pure_data_collection_episodes_nb,
        learn_every_x_steps=learn_every_x_steps,
        learning_steps=learning_steps,
        print_every=print_every,
        plot_every=plot_every,
        save_every=save_every,
        check_performance_every=check_performance_every,
        result_directory_path=result_directory_path,
        score_window_size=100,
        target_score=35.0)

    # Plot the scores
    plot_scores(scores, result_directory_path, pure_data_collection_episodes_nb, show_plot=True)

if __name__ == "__main__":
    main()

