import os
import pygame
from OpenGL.GL import *
import time


def main():

    DISPLAY_INDEX = 1
    TARGET_FPS = 30
    PERIOD = 1.0 / TARGET_FPS  # 33.333 ms
    # ---- FIX 1: Prevent window from hiding when it loses focus ----
    os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'
    
    pygame.init()
    pygame.font.init()

    # Disable vsync (we control timing manually)
    pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 0)

    # Get desktop sizes safely
    try:
        screen_size = pygame.display.get_desktop_sizes()[DISPLAY_INDEX]
    except IndexError:
        print(f"Error: Display index {DISPLAY_INDEX} not found. Defaulting to 0.")
        screen_size = pygame.display.get_desktop_sizes()[0]
        DISPLAY_INDEX = 0 # Fallback to primary if the monitor is disconnected

    pygame.display.set_mode(
        screen_size,
        pygame.OPENGL | pygame.FULLSCREEN | pygame.DOUBLEBUF,
        display=DISPLAY_INDEX
    )

    # ---- OpenGL setup ----
    glViewport(0, 0, screen_size[0], screen_size[1])

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, screen_size[0], screen_size[1], 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glEnable(GL_TEXTURE_2D)

    # ---- Texture ----
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # ---- Font ----
    font = pygame.font.SysFont("Arial", 300)

    counter = 0
    running = True

    next_time = time.perf_counter()

    while running:
        # ---- FIX 2: Pump OS events at the START of the frame ----
        for e in pygame.event.get():
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False
        
        if not running:
            break

        frame_start = time.perf_counter()

        # ---- Render number ----
        text_surface = font.render(str(counter), True, (255, 255, 255), (0, 0, 0))
        text_data = pygame.image.tostring(text_surface, "RGB", True)
        tw, th = text_surface.get_size()

        glBindTexture(GL_TEXTURE_2D, tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tw, th, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, text_data)

        # ---- Draw ----
        glClear(GL_COLOR_BUFFER_BIT)

        x = (screen_size[0] - tw) // 2
        y = (screen_size[1] - th) // 2

        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + tw, y)
        glTexCoord2f(1, 1); glVertex2f(x + tw, y + th)
        glTexCoord2f(0, 1); glVertex2f(x, y + th)
        glEnd()

        pygame.display.flip()

        # ---- Wait until next 30 Hz tick ----
        next_time += PERIOD
        now = time.perf_counter()

        sleep_time = next_time - now
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            # we're late → resync
            next_time = now

        actual_dt = time.perf_counter() - frame_start

        print(f"Displayed: {counter} | Frame: {actual_dt*1000:.2f} ms")

        counter += 1

    pygame.quit()

if __name__ == "__main__":
    main()


