import cv2
import numpy as np
import argparse
import os

def generate_asymmetric_circles_grid(
    cols: int,
    rows: int,
    circle_radius: int,
    spacing: float,
    margin: int,
    output_filename: str,
    background_color: int = 255, # White
    circle_color: int = 0        # Black
):
    """
    Generates an image file with an asymmetric circle grid.

    Args:
        cols: Number of circles in the horizontal direction (wider dimension).
        rows: Number of circles in the vertical direction (narrower dimension).
        circle_radius: Radius of each circle in pixels.
        spacing: Distance between the centers of adjacent circles (horizontally and vertically).
        margin: Border around the grid in pixels.
        output_filename: Path to save the generated image.
        background_color: Grayscale value for the background (0-255).
        circle_color: Grayscale value for the circles (0-255).

    Raises:
        ValueError: If cols and rows are equal (must be asymmetric).
        ValueError: If circle_radius or spacing are not positive.
    """
    if cols == rows:
        raise ValueError(f"Grid must be asymmetric: cols ({cols}) cannot equal rows ({rows}).")
    if circle_radius <= 0:
        raise ValueError("circle_radius must be positive.")
    if spacing <= 2 * circle_radius:
         print(f"Warning: Spacing ({spacing}) is less than or equal to circle diameter "
               f"({2*circle_radius}). Circles might touch or overlap.")
    if spacing <= 0:
        raise ValueError("spacing must be positive.")


    # Calculate the required image dimensions
    # Grid width calculation needs to account for the half-spacing offset
    grid_width = (cols - 1) * spacing + (spacing / 2.0) # Max horizontal extent between centers
    grid_height = (rows - 1) * spacing           # Max vertical extent between centers

    # Image dimensions: grid size + diameter of circles at edges + margin
    img_width = int(np.ceil(grid_width + 2 * circle_radius + 2 * margin))
    img_height = int(np.ceil(grid_height + 2 * circle_radius + 2 * margin))

    # Create a blank canvas (grayscale image)
    image = np.full((img_height, img_width), background_color, dtype=np.uint8)

    # Calculate starting offset to center the grid
    start_x_offset = margin + circle_radius
    start_y_offset = margin + circle_radius

    print(f"Generating {cols}x{rows} grid...")
    print(f"Image dimensions: {img_width}x{img_height} pixels")
    print(f"Circle radius: {circle_radius} pixels")
    print(f"Spacing: {spacing} pixels")

    # Draw the circles
    for r in range(rows):
        for c in range(cols):
            # Calculate center coordinates
            center_y = int(round(start_y_offset + r * spacing))

            # Apply horizontal offset for asymmetric grid (every other row)
            # Offset is applied relative to the *column* spacing
            row_offset = (r % 2) * (spacing / 2.0)
            center_x = int(round(start_x_offset + c * spacing + row_offset))

            # Draw the circle (filled)
            cv2.circle(image, (center_x, center_y), circle_radius, circle_color, -1) # -1 thickness fills the circle

    # Save the image
    try:
        # Ensure directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        success = cv2.imwrite(output_filename, image)
        if success:
            print(f"Successfully saved grid to: {output_filename}")
        else:
            print(f"Error: Failed to save image to {output_filename}")
    except Exception as e:
        print(f"Error saving image: {e}")

    # --- Optional: Display the generated grid ---
    # cv2.imshow(f"Asymmetric Circle Grid ({cols}x{rows})", image)
    # print("Press any key to close the preview window...")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # --- End Optional Display ---

    return image # Return the generated numpy array


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an Asymmetric Circle Grid for OpenCV Calibration")

    parser.add_argument("-c", "--cols", type=int, default=11, help="Number of circles horizontally (wider dimension)")
    parser.add_argument("-r", "--rows", type=int, default=7, help="Number of circles vertically (narrower dimension)")
    parser.add_argument("--radius", type=int, default=15, help="Radius of each circle in pixels")
    parser.add_argument("--spacing", type=float, default=100.0, help="Distance between circle centers in pixels")
    parser.add_argument("--margin", type=int, default=50, help="Margin around the grid in pixels")
    parser.add_argument("-o", "--output", type=str, default="asymmetric_circles_grid.png", help="Output image filename (e.g., grid.png)")

    args = parser.parse_args()

    try:
        generate_asymmetric_circles_grid(
            cols=args.cols,
            rows=args.rows,
            circle_radius=args.radius,
            spacing=args.spacing,
            margin=args.margin,
            output_filename=args.output
        )
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")