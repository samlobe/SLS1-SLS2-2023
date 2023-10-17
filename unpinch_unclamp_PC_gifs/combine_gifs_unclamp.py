#%%
from PIL import Image, ImageSequence
from tqdm import tqdm

# Open the gifs
animation = Image.open("unclamp_just_landscape.gif")
unclamp = Image.open("just_protein_gifs/unclamp_just_protein.gif")

# Create an empty list to store the resized frames
resized_frames = []

# Loop through all frames
print('Resizing frames...')
for frame in tqdm(ImageSequence.Iterator(unclamp)):
    # Calculate the new width while maintaining aspect ratio
    new_width = int(frame.width * (animation.height / frame.height))
    
    # Create a new image with white background
    bg = Image.new("RGB", (new_width, animation.height), (255, 255, 255))
    
    # Resize the frame to match animation height while keeping aspect ratio
    frame = frame.resize((new_width, animation.height))
    
    #convert the mode of images to "RGBA"
    frame = frame.convert("RGBA")
    bg = bg.convert("RGBA")
    
    # composite the frame over the white background
    bg = Image.alpha_composite(bg, frame)
    
    # Append the resized frame to the list
    resized_frames.append(frame)

# Save the resized gif
resized_frames[0].save("unclamp_just_protein_resized.gif", save_all=True, append_images=resized_frames[1:], duration=unclamp.info['duration'], loop=0)

#%%
# Open the resized gif
unclamp = Image.open("unclamp_just_protein_resized.gif")

# Create a new image with the widths of both gifs
result = Image.new("RGB", (animation.width + unclamp.width, animation.height))

# Get the frames from the gifs
animation_frames = animation.seek(0)
unclamp_frames = unclamp.seek(0)

# Loop through all frames and paste them side by side
frames = []
print("Pasting gifs side-by-side ...")
for i in tqdm(range(animation.n_frames)):
    animation.seek(i)
    unclamp.seek(i)
    result.paste(animation, (0, 0))
    result.paste(unclamp, (animation.width, 0))
    frames.append(result.copy())

# Save the result
print("Saving gif...")
frames[0].save("UNCLAMP_MODE.gif", save_all=True, append_images=frames[1:], duration=animation.info['duration'], loop=0)
print("Done!")


