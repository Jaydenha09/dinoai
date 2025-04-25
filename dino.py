# dino.py (TensorFlow V2 Version)

import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By # Import By for modern locators
from selenium.webdriver.chrome.options import Options as ChromeOptions # Renamed for clarity
from selenium.webdriver.chrome.service import Service as ChromeService  # Renamed to avoid potential conflicts
from selenium.common.exceptions import WebDriverException

from time import sleep
from io import BytesIO
import skimage as skimage
from skimage import transform, color, exposure, io
# from skimage.transform import rotate # Not used
import numpy as np
import argparse
from collections import deque
import random
import json
import os
from skimage.util import img_as_uint

# --- TF V2 GPU Memory Configuration (Optional but Recommended) ---
# Prevent TensorFlow from allocating all GPU memory upfront
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"GPU Memory Growth Error: {e}")
# --- End GPU Config ---


# Game Parameters / Hyperparameters
ACTIONS = 3 # number of valid actions (0: do nothing, 1: jump, 2: duck)
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 800. # timesteps to observe before training
EXPLORE = 2500. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 25000 # number of previous transitions to remember
BATCH = 64 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 0.0003 # Learning rate for Adam optimizer
IMG_SIZE = 75 # Size of the processed game frame (IMG_SIZE x IMG_SIZE)
INPUT_FRAMES = 3 # Number of frames stacked together as input

class DinoGame():
    """Handles Chrome Dino game interactions using Selenium."""
    def __init__(self):
        options = ChromeOptions()
        # option.add_argument("--disable-gpu") # Often needed in headless/docker
        options.add_argument("--no-sandbox") # Often needed in docker/linux
        options.add_argument("--disable-dev-shm-usage") # Overcome limited resource problems
        # options.add_argument("--headless") # Uncomment to run without GUI
        options.add_argument("--window-size=800,600") # Set a reasonable window size
        self.driver = webdriver.Chrome(options=options)
        self.observation_space = (IMG_SIZE, IMG_SIZE) # Define observation space dims

    def open(self):
        """Navigates the browser to the Chrome Dino game."""
        self.driver.get('chrome://dino')
        # Wait a bit for the game to load - adjust if needed
        sleep(1)

    def _get_body_element(self):
        """Helper to get the body element for sending keys."""
        return self.driver.find_element(By.TAG_NAME, "body")

    def up(self):
        """Sends the UP arrow key press."""
        self._get_body_element().send_keys(Keys.ARROW_UP)

    def down(self):
        """Sends the DOWN arrow key press."""
        self._get_body_element().send_keys(Keys.ARROW_DOWN)

    def return_nor(self):
        """Placeholder for 'do nothing' action."""
        pass # No key press needed

    def get_crashed(self) -> bool:
        """Checks if the game is in a crashed state."""
        try:
            return self.driver.execute_script("return Runner.instance_.crashed")
        except Exception as e:
            print(f"Error checking crash state: {e}. Assuming not crashed.")
            return False # Safest assumption if script fails

    def get_playing(self) -> bool:
        """Checks if the game is currently playing (not crashed)."""
        return not self.get_crashed()

    def restart(self):
        """Restarts the game via JavaScript."""
        self.driver.execute_script("Runner.instance_.restart()")
        sleep(0.1) # Short pause after restart

    def get_score(self) -> int:
        """Retrieves the current game score."""
        try:
            score_digits = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
            score = ''.join(score_digits)
            return int(score) if score else 0
        except Exception as e:
            print(f"Error getting score: {e}. Returning 0.")
            return 0

    def pause(self):
        """Pauses the game via JavaScript."""
        self.driver.execute_script("Runner.instance_.stop()")

    def resume(self):
        """Resumes the game via JavaScript."""
        self.driver.execute_script("Runner.instance_.play()")

    def start(self):
        """Starts the game by pressing UP."""
        self.up()

    def get_frame(self, count=-1) -> np.ndarray:
        """
        Captures, crops, and preprocesses a game frame.
        Args:
            count (int): Frame counter, used for optional saving (currently disabled).
        Returns:
            np.ndarray: Preprocessed grayscale game frame normalized to [0, 1].
        """
        try:
            # Get details needed for cropping accurately
            canvas_details = self.driver.execute_script("return Runner.instance_.canvas.getBoundingClientRect()")
            actual_width = self.driver.execute_script("return Runner.instance_.canvas.width")
            try:
                dino_width = self.driver.execute_script("return Runner.instance_.tRex.config.WIDTH_DUCK") # Duck width might be more stable
                xPos = self.driver.execute_script("return Runner.instance_.tRex.xPos")
            except Exception:
                 dino_width = 44 # Default T-Rex width approx
                 xPos = 10     # Default T-Rex pos approx

            # Take screenshot
            screenshot = self.driver.get_screenshot_as_png()
            frame_img_raw = skimage.io.imread(BytesIO(screenshot)) # Read raw image (potentially RGBA)

             # Calculate cropping coordinates
            top = int(canvas_details['y'])
            bottom = int(canvas_details['height'] + canvas_details['y'])
            scale_factor = canvas_details['width'] / actual_width
            left = int(canvas_details['x'] + (xPos + dino_width) * scale_factor)
            right = int(canvas_details['x'] + (canvas_details['width'] / 2))

            # Ensure coordinates are valid and within image bounds
            h, w = frame_img_raw.shape[:2] # Get raw image height and width
            top = max(0, top)
            left = max(0, left)
            bottom = min(h, bottom)
            right = min(w, right)

            # Check for invalid crop dimensions *before* cropping
            if top >= bottom or left >= right:
                 print(f"Warning: Invalid crop dimensions calculated (T:{top} B:{bottom} L:{left} R:{right}). Raw H,W: ({h},{w}). Returning blank frame.")
                 # Check potentially problematic values
                 print(f"Debug canvas: {canvas_details}, actual_W:{actual_width}, dino_W:{dino_width}, xPos:{xPos}, scale:{scale_factor}")
                 return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

            # Crop the image
            frame_img_cropped = frame_img_raw[top:bottom, left:right]

            # --- FIX: Convert RGBA to RGB if necessary ---
            processed_img = frame_img_cropped # Start with cropped image

            if processed_img.ndim == 3 and processed_img.shape[-1] == 4:
                # Image is RGBA, convert to RGB
                # print("DEBUG: Converting RGBA -> RGB") # Optional debug print
                # Convert to float [0,1] first, as rgba2rgb expects this or uint8
                processed_img = skimage.util.img_as_float(processed_img)
                processed_img = skimage.color.rgba2rgb(processed_img) # Converts RGBA to RGB

            elif processed_img.ndim == 3 and processed_img.shape[-1] == 3:
                # Image is already RGB, ensure float format
                # print("DEBUG: Image is RGB") # Optional debug print
                processed_img = skimage.util.img_as_float(processed_img)

            elif processed_img.ndim == 2:
                 # Image is already Grayscale, ensure float format
                 # print("DEBUG: Image is Grayscale") # Optional debug print
                 processed_img = skimage.util.img_as_float(processed_img)
            else:
                 # Handle unexpected shapes
                 print(f"Warning: Unexpected image shape after crop: {processed_img.shape}. Returning blank frame.")
                 return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

            # --- Convert to Grayscale (if it wasn't already) ---
            if processed_img.ndim == 3: # Only convert if it has color channels
                 # print("DEBUG: Converting RGB -> Gray") # Optional debug print
                 gray_img = skimage.color.rgb2gray(processed_img) # Input is RGB float [0,1], output is Gray float [0,1]
            else:
                 # print("DEBUG: Image already Gray") # Optional debug print
                 gray_img = processed_img # Already grayscale

            # --- Resize ---
            # print(f"DEBUG: Resizing from {gray_img.shape} to ({IMG_SIZE},{IMG_SIZE})") # Optional debug print
            resized_img = skimage.transform.resize(gray_img, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)

            # Optional: save frame for debugging
            # if count >= 0:
            #   os.makedirs('./frames', exist_ok=True)
            #   # Save the image *before* final type conversion if needed for viewing
            #   io.imsave(f'./frames/frame-{count:06d}.png', skimage.util.img_as_uint(resized_img))

            # Ensure float32 for TensorFlow
            final_frame = resized_img.astype(np.float32)
            # print(f"DEBUG: Final frame shape {final_frame.shape}, dtype {final_frame.dtype}") # Optional debug print
            return final_frame

        except Exception as e:
             print(f"Error capturing/processing frame: {e}. Returning blank frame.")
             import traceback
             traceback.print_exc() # Print full error details for debugging
             # Return a consistent shape on error
             return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)



    def get_reward(self) -> int:
        """Determines the reward based on game state."""
        if self.get_crashed():
            return -1 # Penalty for crashing
        else:
            return 1  # Small reward for surviving

    def take_action(self, action_index: int):
        """Performs an action based on the index."""
        if action_index == 1: # Jump
            self.up()
        elif action_index == 2: # Duck
            self.down()
        # else: action_index == 0: Do nothing

    def close(self):
        """Closes the browser window."""
        if self.driver:
            self.driver.quit()

class DQNAgent():
    """Deep Q-Network Agent."""
    def __init__(self, input_shape, num_actions):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.model = self._build_model()

    def _build_model(self) -> Sequential:
        """Builds the Keras Sequential CNN model."""
        model = Sequential(name="DinoDQN")
        # Input shape: (height, width, channels) -> (IMG_SIZE, IMG_SIZE, INPUT_FRAMES)
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear')) # Output Q-values per action

        # Compile the model
        optimizer = Adam(learning_rate=LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse') # Mean Squared Error for Q-learning
        print("Model Built and Compiled Successfully")
        model.summary() # Print model summary
        return model

    def train(self, game: DinoGame, args: dict):
        """Trains the DQN agent."""
        # Initialize replay memory D
        D = deque(maxlen=REPLAY_MEMORY)

        # Initialize game and get initial state
        game.open()
        game.restart() # Start fresh
        game.start()   # Initial jump to begin game

        # Get initial frame and stack it
        frame = game.get_frame()
        stacked_frames = np.stack([frame] * INPUT_FRAMES, axis=2) # (IMG_SIZE, IMG_SIZE, INPUT_FRAMES)
        # Reshape for model prediction (add batch dimension)
        stacked_frames_batch = stacked_frames.reshape(1, *self.input_shape) #(1, IMG_SIZE, IMG_SIZE, INPUT_FRAMES)

        # Training mode setup
        if args['mode'].lower() == 'run':
            observe_steps = float('inf')  # Run mode: effectively infinite observation
            epsilon = FINAL_EPSILON
            print("--- Running Mode ---")
            try:
                self.model.load_weights("model.h5")
                print("Loaded model weights from model.h5")
            except Exception as e:
                print(f"Could not load weights for run mode: {e}. Starting with random weights.")
        else: # Training mode
            observe_steps = OBSERVATION
            epsilon = INITIAL_EPSILON
            print("--- Training Mode ---")
            if os.path.isfile('model.h5'):
                try:
                    self.model.load_weights("model.h5")
                    print("Loaded existing model weights from model.h5")
                except Exception as e:
                    print(f"Could not load weights: {e}. Starting training from scratch.")
            else:
                print("No existing weights found. Starting training from scratch.")


        t = 0 # Timestep counter
        total_loss = 0.0
        episode_steps = 0

        while True: # Main loop
            loss = 0.0
            q_values = None # Store Q-values for printing
            action_index = 0
            reward = 0
            terminal = False # Reset terminal state check

            # 1. Choose action epsilon-greedily
            if random.random() <= epsilon:
                # print("---------- Random Action ----------")
                action_index = random.randrange(self.num_actions)
            else:
                # Predict Q-values for the current state
                q_values = self.model.predict(stacked_frames_batch, verbose=0)[0] # Get Q-values for the single batch item
                action_index = np.argmax(q_values) # Choose action with highest Q-value

            # Gradually decrease epsilon
            if epsilon > FINAL_EPSILON and t > observe_steps:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
                epsilon = max(FINAL_EPSILON, epsilon) # Ensure epsilon doesn't go below final

            # 2. Take action in the game
            game.take_action(action_index)
            # Allow time for action to register and game to update - adjust as needed
            sleep(0.05) # Small delay

            # 3. Observe next state and reward
            next_frame = game.get_frame(t)
            next_frame_reshaped = next_frame.reshape(1, IMG_SIZE, IMG_SIZE, 1) # Add channel dim

             # Create next stacked frame state
            next_stacked_frames = np.append(next_frame_reshaped, stacked_frames_batch[:, :, :, :INPUT_FRAMES-1], axis=3)

            reward = game.get_reward()
            terminal = game.get_crashed() # Check if the game ended

            # 4. Store transition in replay memory D
            D.append((stacked_frames_batch[0], action_index, reward, next_stacked_frames[0], terminal)) # Store without batch dim

            # 5. Sample minibatch and train if observation phase is over
            if t > observe_steps and len(D) > BATCH:
                # Sample random minibatch
                minibatch = random.sample(D, BATCH)

                # Unzip minibatch
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*minibatch)

                # Convert to numpy arrays for training
                state_batch = np.array(state_batch)             # (BATCH, IMG_SIZE, IMG_SIZE, INPUT_FRAMES)
                action_batch = np.array(action_batch)           # (BATCH,)
                reward_batch = np.array(reward_batch)           # (BATCH,)
                next_state_batch = np.array(next_state_batch)   # (BATCH, IMG_SIZE, IMG_SIZE, INPUT_FRAMES)
                terminal_batch = np.array(terminal_batch)       # (BATCH,)

                # Predict Q-values for current states
                target_q_values = self.model.predict(state_batch, verbose=0)

                # Predict Q-values for next states
                next_q_values = self.model.predict(next_state_batch, verbose=0)

                # Calculate target Q-values using Bellman equation: Q = r + gamma * max(Q_next)
                # For terminal states, the target is just the reward
                max_next_q = np.max(next_q_values, axis=1)
                target_q_values[range(BATCH), action_batch] = reward_batch + GAMMA * max_next_q * (1 - terminal_batch.astype(int))

                # Train the model on the minibatch
                loss = self.model.train_on_batch(state_batch, target_q_values)
                total_loss += loss

            # Update current state
            stacked_frames_batch = next_stacked_frames

            # Increment timestep
            t += 1
            episode_steps += 1

            # Handle game over (crash)
            if terminal:
                print(f"--- Episode End --- Timestep: {t}, Steps in Episode: {episode_steps}, Final Score: {game.get_score()}, Epsilon: {epsilon:.5f}")
                game.restart()
                sleep(0.5) # Pause before starting new game
                game.start()
                episode_steps = 0
                # Reset state after crash (important!)
                frame = game.get_frame()
                stacked_frames = np.stack([frame] * INPUT_FRAMES, axis=2)
                stacked_frames_batch = stacked_frames.reshape(1, *self.input_shape)


            # Save progress periodically during training
            if t > observe_steps and t % 5000 == 0: # Increased save frequency
                print(f"\nSaving model at timestep {t}...")
                self.model.save_weights("model.h5", overwrite=True)
                # Optional: Save model structure (less common now, weights are usually enough if architecture is defined in code)
                # with open("model.json", "w") as outfile:
                #    outfile.write(self.model.to_json())
                print("Model weights saved to model.h5")
                # Optional memory clearing (less critical with TF2 memory growth)
                # print("Clearing memory cache (if possible)...")
                # os.system("sync; echo 1 > /proc/sys/vm/drop_caches") # Linux specific

            # Print status
            if t % 100 == 0: # Print less frequently
                state_str = ""
                if t <= observe_steps:
                    state_str = "observing"
                elif t > observe_steps and epsilon > FINAL_EPSILON:
                    state_str = "exploring"
                else:
                    state_str = "training"

                avg_loss = total_loss / (t - observe_steps) if t > observe_steps else 0
                q_max_str = f"{np.max(q_values):.4f}" if q_values is not None else "N/A"


                print(f"T:{t} | S:{state_str} | E:{epsilon:.5f} | A:{action_index} | R:{reward} | Qmax:{q_max_str} | Loss:{loss:.5f} | AvgLoss:{avg_loss:.5f} | Score:{game.get_score()}", end='\r')

        print("\nTraining finished! (or interrupted)")
        # Final save
        if args['mode'].lower() != 'run':
            print("Saving final model weights...")
            self.model.save_weights("model.h5", overwrite=True)
            print("Final weights saved.")


def main():
    parser = argparse.ArgumentParser(description='Train or Run a DQN agent for the Chrome Dino game.')
    parser.add_argument('-m','--mode', help='Train / Run', required=True, choices=['Train', 'Run', 'train', 'run'])
    args = vars(parser.parse_args())

    # Initialize game environment and agent
    game = None # Initialize game variable
    try:
        game = DinoGame()
        agent = DQNAgent(input_shape=(IMG_SIZE, IMG_SIZE, INPUT_FRAMES), num_actions=ACTIONS)
        agent.train(game, args)
    except KeyboardInterrupt:
        print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\n--- An error occurred ---")
        import traceback
        traceback.print_exc() # Print detailed error information
    finally:
        # Ensure the browser is closed even if errors occur
        if game:
            print("Closing browser...")
            game.close()
            print("Browser closed.")

if __name__ == "__main__":
    # TensorFlow V1 specific session code (removed)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # from keras import backend as K
    # K.set_session(sess)
    main()
