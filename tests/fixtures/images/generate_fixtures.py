import cv2
import numpy as np
import os

def create_halftone_background(width, height, dot_spacing=4):
    """Creates a background with a halftone dot pattern"""
    bg = np.ones((height, width), dtype=np.uint8) * 230  # Off-white paper
    
    # Add halftone dots
    for y in range(0, height, dot_spacing):
        for x in range(0, width, dot_spacing):
            # Modulate dot size slightly for texture
            radius = 1 if (x+y) % 3 == 0 else 0
            if radius > 0:
                cv2.circle(bg, (x, y), radius, (150, 150, 150), -1)
                
    # Add some noise
    noise = np.random.normal(0, 10, (height, width)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Convert to BGR
    return cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

def generate_realistic_newspaper(output_path):
    width, height = 800, 600
    img = create_halftone_background(width, height)
    
    # Add text
    cv2.putText(img, "The Hindu", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 2.5, (30, 30, 30), 4)
    cv2.putText(img, "New Delhi | 12 July", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
    
    cv2.line(img, (50, 180), (750, 180), (30, 30, 30), 2)
    
    cv2.putText(img, "Government launches new policy", (50, 240), cv2.FONT_HERSHEY_COMPLEX, 1.5, (10, 10, 10), 3)
    
    body = "The government today announced a comprehensive new policy aimed at boosting economic growth."
    cv2.putText(img, body, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 1)
    
    cv2.imwrite(output_path, img)
    print(f"Generated {output_path}")

def generate_fake_clipping(output_path):
    width, height = 800, 600
    # Perfectly smooth digital background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    cv2.putText(img, "The Hindu", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 2.5, (0, 0, 0), 4)
    cv2.putText(img, "New Delhi | 12 July", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.line(img, (50, 180), (750, 180), (0, 0, 0), 2)
    
    # Edited headline with a slightly different background color patch to trigger ELA
    # And we'll duplicate a text block to trigger copy-move
    
    # Trigger ELA
    cv2.rectangle(img, (40, 200), (760, 270), (245, 245, 245), -1)
    cv2.putText(img, "ALIENS DISCOVERED ON MARS!!!", (50, 250), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 200), 3)
    
    # Trigger Copy-Move by creating a distinct textured block and copying it
    block = np.zeros((100, 150, 3), dtype=np.uint8)
    cv2.putText(block, "SECRET", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # add noise to block to create keypoints for ORB
    noise = np.random.normal(0, 50, (100, 150, 3)).astype(np.int16)
    block = np.clip(block.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Paste it in two different locations
    img[300:400, 100:250] = block
    img[300:400, 500:650] = block
    
    cv2.imwrite(output_path, img)
    print(f"Generated {output_path}")

def generate_ai_filler(output_path):
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    cv2.putText(img, "AI GENERATED ARTICLE", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 0), 2)
    
    filler = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt."
    cv2.putText(img, filler[:40], (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 1)
    cv2.putText(img, filler[40:], (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 1)
    
    cv2.imwrite(output_path, img)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    generate_realistic_newspaper(os.path.join(os.path.dirname(__file__), "realistic_newspaper.png"))
    generate_fake_clipping(os.path.join(os.path.dirname(__file__), "fake_clipping.png"))
    generate_ai_filler(os.path.join(os.path.dirname(__file__), "ai_filler.png"))
