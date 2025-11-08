# Test Pin Layout Visualization
# Quick test to verify pin numbering is correct

import matplotlib.pyplot as plt
import numpy as np

def test_pin_layout():
    """Test the current pin layout to verify numbering"""
    
    # Duckpin layout: Pin 1 centered in bottom, then 2-3, 4-5-6, 7-8-9-10
    pin_positions = [
        (2.0, 0.5),    # Pin 1 (bottom row, center - only pin)
        (1.5, 1.5),    # Pin 2 (second row, left)
        (2.5, 1.5),    # Pin 3 (second row, right)
        (1.0, 2.5),    # Pin 4 (third row, left)
        (2.0, 2.5),    # Pin 5 (third row, center)
        (3.0, 2.5),    # Pin 6 (third row, right)
        (0.5, 3.5),    # Pin 7 (top row, far left)
        (1.5, 3.5),    # Pin 8 (top row, left)
        (2.5, 3.5),    # Pin 9 (top row, right)
        (3.5, 3.5),    # Pin 10 (top row, far right)
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.0)
    ax.set_aspect('equal')
    
    # Draw pins with numbers
    for i in range(10):
        pin_number = i + 1  # Pin numbers 1-10
        x, y = pin_positions[i]
        
        # Draw pin
        ax.scatter(x, y, c='blue', s=300, marker='o', edgecolors='black', linewidth=2)
        
        # Add pin number
        ax.text(x, y, str(pin_number), ha='center', va='center', 
                fontsize=14, fontweight='bold', color='white')
    
    ax.set_title('Duckpin Pin Layout Test\n(Pin 1 should be at top center)', fontsize=14)
    ax.set_xlabel('Left → Right')
    ax.set_ylabel('Front → Back')
    ax.grid(True, alpha=0.3)
    
    # Add row labels
    ax.text(-0.3, 3.5, 'Back', rotation=90, va='center', fontsize=10)
    ax.text(-0.3, 2.5, 'Row 2', rotation=90, va='center', fontsize=10)
    ax.text(-0.3, 1.5, 'Row 3', rotation=90, va='center', fontsize=10)
    ax.text(-0.3, 0.5, 'Front', rotation=90, va='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    print("Expected duckpin layout:")
    print("7  8  9  10   (top row - 4 pins)")
    print(" 4  5  6      (third row - 3 pins)")
    print("  2  3        (second row - 2 pins)")
    print("   1          (bottom row - 1 pin)")
    print()
    print("Pin 1 is centered in bottom row (only pin)")
    print("If this doesn't match what you see, please let me know what's wrong!")

if __name__ == "__main__":
    test_pin_layout()