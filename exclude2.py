# Reimport necessary libraries
import matplotlib.pyplot as plt

# Define the data for original and replicated vectors
total_boxes = 3  # There are always 3 boxes per vector (0 to 2)
colors = ['#FF6666', '#D3D3D3']  # Red for replicated, Gray for non-replicated

# Setup the plot with 9 rows and 4 columns
fig, axs = plt.subplots(nrows=9, ncols=4, figsize=(12, 24))  # 9 rows and 4 columns
fig.suptitle('Column Replication Across Multiple Rows', fontsize=16, y=1.05)


# Function to draw a smaller rectangle divided into boxes
def draw_small_vector(ax, filled_boxes, total_boxes):
    for i in range(total_boxes):
        color = colors[0] if i < filled_boxes else colors[1]
        rect = plt.Rectangle((i, 0), 1, 1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    ax.set_xlim(0, total_boxes)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


# Loop over the rows and columns to plot the rectangles
for row in range(9):
    for col in range(4):
        if col == 0:
            # The first column always has all boxes filled (3 red)
            filled_boxes =  min((row % 3) + 1, 3)
        elif col == 1:
            # The second rectangle progressively adds more red boxes in all rows
            filled_boxes = min((row % 3) + 1, 3)
        elif col == 2 and row >= 3:
            # The third rectangle updates together with the second in the next 3 rows (3-6)
            filled_boxes = min((row % 3) + 1, 3) if row >= 3 else 0
        elif col == 3 and row >= 6:
            # The fourth rectangle updates together with the second and third in rows 9-12
            filled_boxes = min((row % 3) + 1, 3) if row >= 6 else 0
        else:
            # Other rectangles are gray if they haven't started replication yet
            filled_boxes = 0

        draw_small_vector(axs[row, col], filled_boxes, total_boxes)


# Adjust layout
plt.tight_layout()
plt.show()
