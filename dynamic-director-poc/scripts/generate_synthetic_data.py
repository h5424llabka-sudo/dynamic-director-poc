"""
Synthetic data generator for toddler action recognition model training.

Generates normalized skeleton time-series data for 5 actions + None class,
simulating the output of PoseTimeSeriesBuffer.kt.

Each sample is a [30, 27] tensor:
  - 30 frames (1 second at 30fps equivalent)
  - 27 features: 9 keypoints × 3 (x, y, confidence)

Keypoints (in shoulder-width-normalized space, hip center = origin):
  0: Nose, 1: L-Shoulder, 2: R-Shoulder, 3: L-Elbow, 4: R-Elbow,
  5: L-Wrist, 6: R-Wrist, 7: L-Hip, 8: R-Hip

Actions:
  0: None, 1: Banzai, 2: Pointing, 3: Waving, 4: Throwing, 5: Clapping
"""

import numpy as np
import os

# Constants matching PoseTimeSeriesBuffer.kt
NUM_FRAMES = 30
NUM_KEYPOINTS = 9
CHANNELS = 3  # x, y, confidence
FEATURE_DIM = NUM_KEYPOINTS * CHANNELS  # 27

# Action labels
ACTION_NONE = 0
ACTION_BANZAI = 1
ACTION_POINTING = 2
ACTION_WAVING = 3
ACTION_THROWING = 4
ACTION_CLAPPING = 5
ACTION_NAMES = ["None", "Banzai", "Pointing", "Waving", "Throwing", "Clapping"]
NUM_ACTIONS = len(ACTION_NAMES)

# Default toddler proportions (in shoulder-width units, hip center = origin)
# These represent a typical 2-3 year old child's skeleton proportions
TODDLER_BASE_SKELETON = {
    # Keypoint: (x, y) in normalized space
    "nose":        ( 0.00, -2.50),  # Head is proportionally large
    "l_shoulder":  (-0.50, -1.80),
    "r_shoulder":  ( 0.50, -1.80),
    "l_elbow":     (-0.80, -1.20),
    "r_elbow":     ( 0.80, -1.20),
    "l_wrist":     (-0.90, -0.60),
    "r_wrist":     ( 0.90, -0.60),
    "l_hip":       (-0.30,  0.00),  # Origin
    "r_hip":       ( 0.30,  0.00),
}

KEYPOINT_ORDER = [
    "nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
    "l_wrist", "r_wrist", "l_hip", "r_hip"
]


def get_base_pose():
    """Get the base toddler skeleton as a flat array [27]."""
    pose = np.zeros(FEATURE_DIM, dtype=np.float32)
    for i, name in enumerate(KEYPOINT_ORDER):
        x, y = TODDLER_BASE_SKELETON[name]
        pose[i * CHANNELS] = x
        pose[i * CHANNELS + 1] = y
        pose[i * CHANNELS + 2] = 0.95  # High confidence
    return pose


def add_noise(sequence, position_std=0.05, confidence_drop_prob=0.05):
    """Add Gaussian noise and random confidence drops to a sequence."""
    noisy = sequence.copy()
    for t in range(NUM_FRAMES):
        for kp in range(NUM_KEYPOINTS):
            base = kp * CHANNELS
            # Position noise
            noisy[t, base] += np.random.normal(0, position_std)
            noisy[t, base + 1] += np.random.normal(0, position_std)
            # Random confidence drops
            if np.random.random() < confidence_drop_prob:
                noisy[t, base + 2] = np.random.uniform(0.1, 0.4)
    return noisy


def add_fps_jitter(sequence, drop_prob=0.1):
    """Simulate FPS drops by zeroing out random frames (will be interpolated)."""
    jittered = sequence.copy()
    for t in range(1, NUM_FRAMES - 1):  # Keep first and last
        if np.random.random() < drop_prob:
            jittered[t] = jittered[t - 1]  # Duplicate previous frame
    return jittered


def generate_none(n_samples):
    """Generate 'None' class: natural standing/idle poses with minor movements."""
    samples = []
    for _ in range(n_samples):
        base = get_base_pose()
        sequence = np.tile(base, (NUM_FRAMES, 1))
        # Add small natural sway
        for t in range(NUM_FRAMES):
            sway = np.sin(2 * np.pi * t / NUM_FRAMES * np.random.uniform(0.5, 2)) * 0.05
            for kp in range(NUM_KEYPOINTS):
                sequence[t, kp * CHANNELS] += sway
        sequence = add_noise(sequence, position_std=0.03)
        sequence = add_fps_jitter(sequence)
        samples.append(sequence)
    return np.array(samples)


def generate_banzai(n_samples):
    """Generate 'Banzai': both hands raised above head."""
    samples = []
    for _ in range(n_samples):
        base = get_base_pose()
        sequence = np.tile(base, (NUM_FRAMES, 1))
        
        # Raise both arms over ~10 frames, hold for remaining
        raise_start = np.random.randint(0, 5)
        raise_duration = np.random.randint(5, 12)
        
        # Target positions for raised wrists and elbows
        target_l_elbow = (-0.45, -2.60)
        target_r_elbow = ( 0.45, -2.60)
        target_l_wrist = (-0.40 + np.random.uniform(-0.2, 0.2), -3.00 + np.random.uniform(-0.3, 0.1))
        target_r_wrist = ( 0.40 + np.random.uniform(-0.2, 0.2), -3.00 + np.random.uniform(-0.3, 0.1))
        
        for t in range(NUM_FRAMES):
            progress = np.clip((t - raise_start) / max(raise_duration, 1), 0, 1)
            # Smooth interpolation
            p = 0.5 * (1 - np.cos(np.pi * progress))
            
            # Left elbow
            sequence[t, 3 * CHANNELS] = base[3 * CHANNELS] + (target_l_elbow[0] - base[3 * CHANNELS]) * p
            sequence[t, 3 * CHANNELS + 1] = base[3 * CHANNELS + 1] + (target_l_elbow[1] - base[3 * CHANNELS + 1]) * p
            # Right elbow
            sequence[t, 4 * CHANNELS] = base[4 * CHANNELS] + (target_r_elbow[0] - base[4 * CHANNELS]) * p
            sequence[t, 4 * CHANNELS + 1] = base[4 * CHANNELS + 1] + (target_r_elbow[1] - base[4 * CHANNELS + 1]) * p
            # Left wrist
            sequence[t, 5 * CHANNELS] = base[5 * CHANNELS] + (target_l_wrist[0] - base[5 * CHANNELS]) * p
            sequence[t, 5 * CHANNELS + 1] = base[5 * CHANNELS + 1] + (target_l_wrist[1] - base[5 * CHANNELS + 1]) * p
            # Right wrist
            sequence[t, 6 * CHANNELS] = base[6 * CHANNELS] + (target_r_wrist[0] - base[6 * CHANNELS]) * p
            sequence[t, 6 * CHANNELS + 1] = base[6 * CHANNELS + 1] + (target_r_wrist[1] - base[6 * CHANNELS + 1]) * p
        
        sequence = add_noise(sequence)
        sequence = add_fps_jitter(sequence)
        samples.append(sequence)
    return np.array(samples)


def generate_pointing(n_samples):
    """Generate 'Pointing': one arm extended, other retracted."""
    samples = []
    for _ in range(n_samples):
        base = get_base_pose()
        sequence = np.tile(base, (NUM_FRAMES, 1))
        
        # Randomly choose left or right arm
        is_left = np.random.random() > 0.5
        extend_start = np.random.randint(0, 8)
        extend_duration = np.random.randint(5, 10)
        
        # Direction of pointing (angle from horizontal)
        angle = np.random.uniform(-0.3, 0.3)  # Slight up/down variation
        arm_length = np.random.uniform(2.0, 3.0)  # Extended arm length
        
        if is_left:
            target_elbow = (-1.2, -1.80 + angle * 0.5)
            target_wrist = (-arm_length, -1.80 + angle)
            # Retract right arm
            retract_elbow = (0.5, -1.10)
            retract_wrist = (0.4, -0.50)
        else:
            target_elbow = (1.2, -1.80 + angle * 0.5)
            target_wrist = (arm_length, -1.80 + angle)
            retract_elbow = (-0.5, -1.10)
            retract_wrist = (-0.4, -0.50)
        
        for t in range(NUM_FRAMES):
            progress = np.clip((t - extend_start) / max(extend_duration, 1), 0, 1)
            p = 0.5 * (1 - np.cos(np.pi * progress))
            
            if is_left:
                # Extend left
                sequence[t, 3 * CHANNELS] = base[3 * CHANNELS] + (target_elbow[0] - base[3 * CHANNELS]) * p
                sequence[t, 3 * CHANNELS + 1] = base[3 * CHANNELS + 1] + (target_elbow[1] - base[3 * CHANNELS + 1]) * p
                sequence[t, 5 * CHANNELS] = base[5 * CHANNELS] + (target_wrist[0] - base[5 * CHANNELS]) * p
                sequence[t, 5 * CHANNELS + 1] = base[5 * CHANNELS + 1] + (target_wrist[1] - base[5 * CHANNELS + 1]) * p
                # Retract right
                sequence[t, 4 * CHANNELS] = base[4 * CHANNELS] + (retract_elbow[0] - base[4 * CHANNELS]) * p
                sequence[t, 4 * CHANNELS + 1] = base[4 * CHANNELS + 1] + (retract_elbow[1] - base[4 * CHANNELS + 1]) * p
                sequence[t, 6 * CHANNELS] = base[6 * CHANNELS] + (retract_wrist[0] - base[6 * CHANNELS]) * p
                sequence[t, 6 * CHANNELS + 1] = base[6 * CHANNELS + 1] + (retract_wrist[1] - base[6 * CHANNELS + 1]) * p
            else:
                # Extend right
                sequence[t, 4 * CHANNELS] = base[4 * CHANNELS] + (target_elbow[0] - base[4 * CHANNELS]) * p
                sequence[t, 4 * CHANNELS + 1] = base[4 * CHANNELS + 1] + (target_elbow[1] - base[4 * CHANNELS + 1]) * p
                sequence[t, 6 * CHANNELS] = base[6 * CHANNELS] + (target_wrist[0] - base[6 * CHANNELS]) * p
                sequence[t, 6 * CHANNELS + 1] = base[6 * CHANNELS + 1] + (target_wrist[1] - base[6 * CHANNELS + 1]) * p
                # Retract left
                sequence[t, 3 * CHANNELS] = base[3 * CHANNELS] + (retract_elbow[0] - base[3 * CHANNELS]) * p
                sequence[t, 3 * CHANNELS + 1] = base[3 * CHANNELS + 1] + (retract_elbow[1] - base[3 * CHANNELS + 1]) * p
                sequence[t, 5 * CHANNELS] = base[5 * CHANNELS] + (retract_wrist[0] - base[5 * CHANNELS]) * p
                sequence[t, 5 * CHANNELS + 1] = base[5 * CHANNELS + 1] + (retract_wrist[1] - base[5 * CHANNELS + 1]) * p
        
        sequence = add_noise(sequence)
        sequence = add_fps_jitter(sequence)
        samples.append(sequence)
    return np.array(samples)


def generate_waving(n_samples):
    """Generate 'Waving': hand raised with left-right oscillation."""
    samples = []
    for _ in range(n_samples):
        base = get_base_pose()
        sequence = np.tile(base, (NUM_FRAMES, 1))
        
        is_left = np.random.random() > 0.5
        wrist_idx = 5 if is_left else 6
        elbow_idx = 3 if is_left else 4
        
        freq = np.random.uniform(2.0, 5.0)  # Oscillation frequency (Hz)
        amplitude = np.random.uniform(0.5, 1.5)  # X-axis amplitude
        
        for t in range(NUM_FRAMES):
            time_sec = t / NUM_FRAMES  # 0 to 1 second
            
            # Raise wrist above head
            sequence[t, wrist_idx * CHANNELS + 1] = -3.0 + np.random.uniform(-0.2, 0.2)
            sequence[t, elbow_idx * CHANNELS + 1] = -2.5 + np.random.uniform(-0.1, 0.1)
            
            # Oscillate X position
            wave_x = amplitude * np.sin(2 * np.pi * freq * time_sec)
            side = -1 if is_left else 1
            sequence[t, wrist_idx * CHANNELS] = side * 0.3 + wave_x
            sequence[t, elbow_idx * CHANNELS] = side * 0.4 + wave_x * 0.5
        
        sequence = add_noise(sequence)
        sequence = add_fps_jitter(sequence)
        samples.append(sequence)
    return np.array(samples)


def generate_throwing(n_samples):
    """Generate 'Throwing': rapid downward arm sweep (wind-up → follow-through)."""
    samples = []
    for _ in range(n_samples):
        base = get_base_pose()
        sequence = np.tile(base, (NUM_FRAMES, 1))
        
        is_left = np.random.random() > 0.5
        wrist_idx = 5 if is_left else 6
        elbow_idx = 3 if is_left else 4
        side = -1 if is_left else 1
        
        # Phase timing
        windup_end = np.random.randint(8, 15)
        throw_duration = np.random.randint(3, 8)
        
        for t in range(NUM_FRAMES):
            if t < windup_end:
                # Wind-up: raise arm above shoulder
                progress = t / windup_end
                y_wrist = -0.60 + (-2.80 - (-0.60)) * progress
                y_elbow = -1.20 + (-2.20 - (-1.20)) * progress
            elif t < windup_end + throw_duration:
                # Throw: rapid downward motion
                progress = (t - windup_end) / throw_duration
                p = progress ** 0.5  # Accelerating motion
                y_wrist = -2.80 + (1.00 - (-2.80)) * p
                y_elbow = -2.20 + (-0.50 - (-2.20)) * p
            else:
                # Follow-through: hold low position
                y_wrist = 1.00 + np.random.uniform(-0.1, 0.1)
                y_elbow = -0.50 + np.random.uniform(-0.1, 0.1)
            
            sequence[t, wrist_idx * CHANNELS] = side * 0.9
            sequence[t, wrist_idx * CHANNELS + 1] = y_wrist
            sequence[t, elbow_idx * CHANNELS] = side * 0.7
            sequence[t, elbow_idx * CHANNELS + 1] = y_elbow
        
        sequence = add_noise(sequence)
        sequence = add_fps_jitter(sequence)
        samples.append(sequence)
    return np.array(samples)


def generate_clapping(n_samples):
    """Generate 'Clapping': both wrists approaching and separating at chest level."""
    samples = []
    for _ in range(n_samples):
        base = get_base_pose()
        sequence = np.tile(base, (NUM_FRAMES, 1))
        
        freq = np.random.uniform(3.0, 6.0)  # Clapping frequency
        chest_y = np.random.uniform(-1.2, -0.8)  # Chest level
        
        for t in range(NUM_FRAMES):
            time_sec = t / NUM_FRAMES
            
            # Wrists oscillate together and apart
            separation = 0.15 + 0.6 * abs(np.sin(2 * np.pi * freq * time_sec))
            
            # Both at chest level
            sequence[t, 5 * CHANNELS] = -separation  # L-Wrist X
            sequence[t, 5 * CHANNELS + 1] = chest_y   # L-Wrist Y
            sequence[t, 6 * CHANNELS] = separation     # R-Wrist X
            sequence[t, 6 * CHANNELS + 1] = chest_y    # R-Wrist Y
            
            # Elbows follow
            sequence[t, 3 * CHANNELS] = -(separation + 0.3)
            sequence[t, 3 * CHANNELS + 1] = chest_y + 0.3
            sequence[t, 4 * CHANNELS] = separation + 0.3
            sequence[t, 4 * CHANNELS + 1] = chest_y + 0.3
        
        sequence = add_noise(sequence)
        sequence = add_fps_jitter(sequence)
        samples.append(sequence)
    return np.array(samples)


def generate_dataset(samples_per_class=500, output_dir="data"):
    """Generate the full synthetic dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    generators = [
        (ACTION_NONE, generate_none),
        (ACTION_BANZAI, generate_banzai),
        (ACTION_POINTING, generate_pointing),
        (ACTION_WAVING, generate_waving),
        (ACTION_THROWING, generate_throwing),
        (ACTION_CLAPPING, generate_clapping),
    ]
    
    all_data = []
    all_labels = []
    
    for action_id, generator in generators:
        print(f"Generating {samples_per_class} samples for '{ACTION_NAMES[action_id]}'...")
        data = generator(samples_per_class)
        labels = np.full(samples_per_class, action_id, dtype=np.int32)
        all_data.append(data)
        all_labels.append(labels)
    
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # Shuffle
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    # Split train/val/test (70/15/15)
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    
    # Save
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "X_val.npy"), X_val)
    np.save(os.path.join(output_dir, "y_val.npy"), y_val)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test)
    
    print(f"\nDataset saved to '{output_dir}/':")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Shape: {X_train.shape[1:]} (frames, features)")
    print(f"  Classes: {ACTION_NAMES}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    np.random.seed(42)
    generate_dataset(samples_per_class=500, output_dir="data")
