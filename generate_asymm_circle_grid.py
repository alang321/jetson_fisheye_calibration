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
    scale: float = 1.0, # Resolution multiplier
    background_color: int = 255, # White
    circle_color: int = 0        # Black
):
    """
    Generates an image file with an asymmetric circle grid, optionally scaled.

    Args:
        cols: Number of circles in the horizontal direction (wider dimension).
        rows: Number of circles in the vertical direction (narrower dimension).
        circle_radius: Base radius of each circle in pixels (before scaling).
        spacing: Base distance between the centers of adjacent circles (before scaling).
        margin: Base border around the grid in pixels (before scaling).
        output_filename: Path to save the generated image.
        scale: Resolution multiplier. Scales radius, spacing, and margin. Defaults to 1.0.
        background_color: Grayscale value for the background (0-255).
        circle_color: Grayscale value for the circles (0-255).

    Raises:
        ValueError: If cols and rows are equal (must be asymmetric).
        ValueError: If base circle_radius or spacing are not positive.
        ValueError: If scale is not positive.
    """
    if cols == rows:
        raise ValueError(f"Grid must be asymmetric: cols ({cols}) cannot equal rows ({rows}).")
    if circle_radius <= 0:
        raise ValueError("Base circle_radius must be positive.")
    if spacing <= 0:
        raise ValueError("Base spacing must be positive.")
    if scale <= 0:
        raise ValueError("scale must be positive.")

    # Apply the scale factor
    scaled_radius = int(round(circle_radius * scale))
    scaled_spacing = spacing * scale
    scaled_margin = int(round(margin * scale))

    # Ensure scaled radius is at least 1 pixel after rounding
    if scaled_radius < 1:
         print(f"Warning: Scaled radius ({circle_radius} * {scale} = {circle_radius*scale}) rounded to {scaled_radius}. Setting to minimum 1 pixel.")
         scaled_radius = 1

    # Warning for potential overlap (using scaled values)
    if scaled_spacing <= 2 * scaled_radius:
         print(f"Warning: Scaled spacing ({scaled_spacing:.2f}) is less than or equal to scaled circle diameter "
               f"({2*scaled_radius}). Circles might touch or overlap.")


    # Calculate the required image dimensions using scaled values
    # Grid width calculation needs to account for the half-spacing offset
    grid_width = (cols - 1) * scaled_spacing + (scaled_spacing / 2.0) # Max horizontal extent between centers
    grid_height = (rows - 1) * scaled_spacing           # Max vertical extent between centers

    # Image dimensions: grid size + diameter of circles at edges + margin
    # Use ceil to ensure the grid fully fits
    img_width = int(np.ceil(grid_width + 2 * scaled_radius + 2 * scaled_margin))
    img_height = int(np.ceil(grid_height + 2 * scaled_radius + 2 * scaled_margin))

    # Create a blank canvas (grayscale image)
    image = np.full((img_height, img_width), background_color, dtype=np.uint8)

    # Calculate starting offset to center the grid using scaled values
    start_x_offset = scaled_margin + scaled_radius
    start_y_offset = scaled_margin + scaled_radius

    print(f"Generating {cols}x{rows} grid with scale factor {scale}...")
    print(f"Base parameters: radius={circle_radius}, spacing={spacing}, margin={margin}")
    print(f"Scaled parameters: radius={scaled_radius}, spacing={scaled_spacing:.2f}, margin={scaled_margin}")
    print(f"Output image dimensions: {img_width}x{img_height} pixels")


    # Draw the circles using scaled values
    for r in range(rows):
        for c in range(cols):
            # Calculate center coordinates using scaled spacing and offsets
            center_y = int(round(start_y_offset + r * scaled_spacing))

            # Apply horizontal offset for asymmetric grid (every other row)
            row_offset = (r % 2) * (scaled_spacing / 2.0)
            center_x = int(round(start_x_offset + c * scaled_spacing + row_offset))

            # Draw the circle (filled) using scaled radius
            cv2.circle(image, (center_x, center_y), scaled_radius, circle_color, -1) # -1 thickness fills the circle

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
    # cv2.imshow(f"Asymmetric Circle Grid ({cols}x{rows}, scale={scale})", image)
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
    parser.add_argument("--radius", type=int, default=15, help="Base radius of each circle in pixels (before scaling)")
    parser.add_argument("--spacing", type=float, default=100.0, help="Base distance between circle centers in pixels (before scaling)")
    parser.add_argument("--margin", type=int, default=50, help="Base margin around the grid in pixels (before scaling)")
    parser.add_argument("--scale", type=float, default=10.0, help="Resolution multiplier. Scales radius, spacing, and margin.")
    parser.add_argument("-o", "--output", type=str, default="asymmetric_circles_grid.png", help="Output image filename (e.g., grid.png)")

    args = parser.parse_args()

    try:
        generate_asymmetric_circles_grid(
            cols=args.cols,
            rows=args.rows,
            circle_radius=args.radius,
            spacing=args.spacing,
            margin=args.margin,
            scale=args.scale, # Pass the scale argument
            output_filename=args.output
        )
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")