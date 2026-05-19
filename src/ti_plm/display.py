"""
This module provides structures to help display content on external monitors via HDMI/DP. It is based on pygame so
make sure you have installed the latest version in your python environment.
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
from glob import glob
import pathlib
import logging
import param
from PIL import Image as PILImage

try:
    import pygame as pg
    from pygame._sdl2 import Window as PGWindow, Texture, Renderer
    from PIL.Image import Image
    from screeninfo.screeninfo import get_monitors
except ImportError as e:
    msg = '`ti_plm.display` module requires `pygame`, `pillow`, and `screeninfo` to be installed. Please install these with pip/conda and try again.'
    try:
        e.add_note(msg)
    except:
        e.msg = f'{e.msg}\n{msg}'
    raise e

from . import TIPLMException

log = logging.getLogger()

IMAGE_EXTENSIONS = ('.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff')


class TIPLMDisplayException(TIPLMException):
    pass


class EventLoopExit(TIPLMDisplayException):
    pass


class EventLoop(param.Parameterized):

    fps = param.Integer(default=30, doc='Target FPS for event loop')
    
    enable_escape_exit = param.Boolean(default=True, doc='Enable/disable using the `ESC` key to exit the event loop.')
    
    init_callback = param.Callable(doc='Custom callback function that will be invoked before event loop starts.')
    
    loop_callback = param.Callable(doc='Custom callback function that will be invoked at the beginning of each loop.')
    
    keydown_callback = param.Callable(doc='Custom callback function that will be invoked when a key is pressed. The `key` value from the event object will be passed to the function.')
    
    exit_callback = param.Callable(doc='Custom callback function that will be invoked after event loop exits.')
    
    def __init__(self, **params):
        pg.init()
        self._clock = pg.time.Clock()
        super().__init__(**params)
    
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args, **kwargs):
        self.stop()
    
    def start(self):
        """Optionally overload this function in subclass"""
    
    def stop(self):
        """Optionally overload this function in subclass"""
    
    def update(self):
        """Optionally overload this function in subclass"""

    def draw(self):
        """Optionally overload this function in subclass"""
    
    def loop(self):
        """Run a single event loop, calling update() and draw() once"""
        
        if callable(self.loop_callback):
            self.loop_callback()
        
        for event in pg.event.get():
            if event.type == pg.QUIT:
                raise EventLoopExit
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE and self.enable_escape_exit:
                    raise EventLoopExit
                if callable(self.keydown_callback):
                    self.keydown_callback(event.key)
                if hasattr(self, 'on_keydown'):
                    self.on_keydown(event.key)

        self.update()
        self.draw()
        self._clock.tick(self.fps)
    
    def run(self):
        """Run the event loop until something raises EventLoopExit.
        This will happen automatically if the pg.QUIT event is received or the ESC key is pressed.
        """
        
        if callable(self.init_callback):
            self.init_callback()
        
        while True:
            try:
                self.loop()
            except EventLoopExit:
                break
        
        if callable(self.exit_callback):
            self.exit_callback()
            

class Window(EventLoop):
    
    enable_fullscreen_toggle = param.Boolean(default=True, doc='Enable/disable fullscreen toggle using the `f` key.')
    
    hide_mouse_fullscreen = param.Boolean(default=True, doc='Whether or not to hide the mouse when the window is fullscreen.')
    
    fullscreen = param.Boolean(default=False, doc='Enable/disable fullscreen')
    
    monitor = param.Integer(default=-1, doc='Select the monitor index where the window should be. This can also be updated after the window is created to move the window programmatically.')
    
    def __init__(self, **params):
        # init monitors array and set bounds on monitor index param (to allow negative indexing)
        self._monitors = get_monitors()
        self.param.monitor.bounds = [-len(self._monitors), len(self._monitors) - 1]
        self._window = None
        self._renderer = None
        super().__init__(**params)
    
    @param.depends('fullscreen', 'monitor', watch=True)
    def _update_window(self):
        """Internal function that runs automatically any time fullscreen or monitor params are changed.
        
        It positions the window on the desired monitor and enables/disables fullscreen.
        """
        mon = self._monitors[self.monitor]
        self._window.position = mon.x, mon.y
        if self.fullscreen:
            self._window.set_fullscreen(True)
            if self.hide_mouse_fullscreen:
                pg.mouse.set_visible(False)
        else:
            self._window.set_windowed()
            pg.mouse.set_visible(True)
    
    def start(self):
        """Create pygame window"""
        super().start()
        if self._window is None:
            self._window = PGWindow(opengl=True, resizable=True)
            self._renderer = Renderer(self._window, vsync=True)
            self._update_window()
            self._window.focus()
    
    def stop(self):
        """Destroy pygame window"""
        if self._window is not None:
            self._window.destroy()
            self._renderer = None
        self._window = None
        super().stop()


class ImageWindow(Window):
    
    index = param.Integer(doc='Image index to display')
    
    def __init__(self, **params):
        self._imgs = []
        super().__init__(**params)

    @staticmethod
    def _pad_pil_image_if_needed(img: Image) -> Image:
        if img.size == (2712, 1600):
            padded = PILImage.new(img.mode, (2716, 1600), 0)
            padded.paste(img, (2, 0))
            return padded
        return img

    @staticmethod
    def _pad_surface_if_needed(img: pg.Surface) -> pg.Surface:
        if img.get_size() == (2712, 1600):
            padded = pg.Surface((2716, 1600))
            padded.fill((0, 0, 0))
            padded.blit(img, (2, 0))
            return padded
        return img
    
    def load(self, img: str | pathlib.Path | Image | pg.Surface, recursive: bool = False):
        """
        Load an image or series of images for displaying in this window. Image are appended to the 
        current list of images. Call .clear() to remove all images from the list.

        Args:
            img (str | pathlib.Path | Image | pg.Surface): Input image path, glob string, PIL Image, or pygame Surface
            recursive (bool, optional): Whether or not recursive globing is used on input glob string. Defaults to False.

        Raises:
            TIPLMDisplayException: Error loading requested image(s)
        """
        if isinstance(img, (str, pathlib.Path)):
            p = pathlib.Path(img)
            if p.is_dir():
                paths = list((p.rglob if recursive else p.glob)('*'))
            else:
                paths = [pathlib.Path(p) for p in glob(str(img), recursive=recursive)]
            self._imgs.extend([p for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS])
            if len(self._imgs) == 0:
                raise TIPLMDisplayException(f'No images found for input path "{img}"')
        elif isinstance(img, Image):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = self._pad_pil_image_if_needed(img)
            self._imgs.append(pg.image.frombytes(img.tobytes(), img.size, img.mode))
        elif isinstance(img, pg.Surface):
            self._imgs.append(self._pad_surface_if_needed(img))
        else:
            raise TIPLMDisplayException('Error loading image. Type must be str, pathlib.Path, PIL.Image.Image, or pg.Surface.')
        
        # Reset index to 0, but discard event because we'll manually trigger it later. This ensures event is triggered even if index is already 0.
        with param.discard_events(self):
            self.index = 0
        
        # Manually trigger index param to update texture
        self.param.trigger('index')
        
        return self
    
    def clear(self):
        """Clear image list. This will also clear the image window."""
        self._imgs.clear()
        with param.discard_events(self):
            self.index = 0
        self.param.trigger('index')
    
    @param.depends('index', watch=True)
    def _update_texture(self):
        """Update texture and render image defined by `index` param
        """
        self._renderer.clear()
        
        if len(self._imgs) > 0:
            img = self._imgs[self.index % len(self._imgs)]
        
            if isinstance(img, (str, pathlib.Path)):
                img = pg.image.load(img)

            if isinstance(img, pg.Surface):
                img = self._pad_surface_if_needed(img)
        
            tex = Texture.from_surface(self._renderer, img)
            self._renderer.blit(tex, tex.get_rect())
            
        self._renderer.present()

    def next(self):
        """Display next image if multiple images have been loaded"""
        if len(self._imgs) > 0:
            self.index = (self.index + 1) % len(self._imgs)
        return self
    
    def previous(self):
        """Display previous image if multiple images have been loaded"""
        if len(self._imgs) > 0:
            self.index = (self.index - 1) % len(self._imgs)
        return self
    
    def on_keydown(self, key):
        """Handle keydown event to trigger next/previous image"""
        if key == pg.K_TAB or key == pg.K_RIGHT or key == pg.K_SPACE:
            self.next()
        elif key == pg.K_LEFT:
            self.previous()



class ImageWindow_Open_GL(Window):
    
    index = param.Integer(doc='Image index to display')
    
    def __init__(self, **params):
        self._imgs = []
        self._gl = None
        self._gl_tex_id = None
        self._window_size = (0, 0)
        super().__init__(**params)

    def _import_gl(self):
        if self._gl is None:
            try:
                from OpenGL import GL
            except ImportError as exc:
                raise TIPLMDisplayException(
                    'ImageWindow_Open_GL requires PyOpenGL. Install with `pip install PyOpenGL`.'
                ) from exc
            self._gl = GL
        return self._gl

    def _normalize_monitor_index(self) -> int:
        idx = int(self.monitor)
        n_mon = int(len(self._monitors))
        if idx < 0:
            idx += n_mon
        return max(0, min(n_mon - 1, idx))

    def _setup_gl_state(self):
        GL = self._import_gl()
        width, height = self._window_size
        width = max(1, int(width))
        height = max(1, int(height))

        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, width, height, 0, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

        if self._gl_tex_id is None:
            self._gl_tex_id = int(GL.glGenTextures(1))

        GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._gl_tex_id))
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

    def _recreate_gl_window(self):
        mon = self._monitors[self.monitor]
        self._window_size = (int(mon.width), int(mon.height))

        os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'
        if not bool(self.fullscreen):
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{int(mon.x)},{int(mon.y)}"

        try:
            pg.display.gl_set_attribute(pg.GL_DOUBLEBUFFER, 1)
            
        except Exception:
            pass

        flags = pg.OPENGL | pg.DOUBLEBUF
        if bool(self.fullscreen):
            flags |= pg.FULLSCREEN

        pg.display.set_mode(
            self._window_size,
            flags,
            display=self._normalize_monitor_index(),
        )

        if bool(self.fullscreen) and bool(self.hide_mouse_fullscreen):
            pg.mouse.set_visible(False)
        else:
            pg.mouse.set_visible(True)

        self._setup_gl_state()

    @param.depends('fullscreen', 'monitor', watch=True)
    def _update_window(self):
        if self._window is None:
            return
        self._recreate_gl_window()

    def start(self):
        """Create pygame OpenGL window"""
        EventLoop.start(self)
        if self._window is None:
            self._window = True
            self._renderer = None
            self._recreate_gl_window()
            self.param.trigger('index')

    def stop(self):
        """Destroy pygame OpenGL window"""
        if self._window is not None:
            if self._gl is not None and self._gl_tex_id is not None:
                try:
                    self._gl.glDeleteTextures([int(self._gl_tex_id)])
                except Exception:
                    pass
            self._gl_tex_id = None
            pg.display.quit()
            self._renderer = None
        self._window = None
        EventLoop.stop(self)

    @staticmethod
    def _pad_pil_image_if_needed(img: Image) -> Image:
        if img.size == (2712, 1600):
            padded = PILImage.new(img.mode, (2716, 1600), 0)
            padded.paste(img, (2, 0))
            return padded
        return img

    @staticmethod
    def _pad_surface_if_needed(img: pg.Surface) -> pg.Surface:
        if img.get_size() == (2712, 1600):
            padded = pg.Surface((2716, 1600))
            padded.fill((0, 0, 0))
            padded.blit(img, (2, 0))
            return padded
        return img
    
    def load(self, img: str | pathlib.Path | Image | pg.Surface, recursive: bool = False):
        """Load an image or series of images for displaying in this window. Image are appended to the 
        current list of images. Call .clear() to remove all images from the list.

        Args:
            img (str | pathlib.Path | Image | pg.Surface): Input image path, glob string, PIL Image, or pygame Surface
            recursive (bool, optional): Whether or not recursive globing is used on input glob string. Defaults to False.

        Raises:
            TIPLMDisplayException: Error loading requested image(s)
        """
        if isinstance(img, (str, pathlib.Path)):
            p = pathlib.Path(img)
            if p.is_dir():
                paths = list((p.rglob if recursive else p.glob)('*'))
            else:
                paths = [pathlib.Path(p) for p in glob(str(img), recursive=recursive)]
            self._imgs.extend([p for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS])
            if len(self._imgs) == 0:
                raise TIPLMDisplayException(f'No images found for input path "{img}"')
        elif isinstance(img, Image):
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = self._pad_pil_image_if_needed(img)
            self._imgs.append(pg.image.frombytes(img.tobytes(), img.size, img.mode))
        elif isinstance(img, pg.Surface):
            self._imgs.append(self._pad_surface_if_needed(img))
        else:
            raise TIPLMDisplayException('Error loading image. Type must be str, pathlib.Path, PIL.Image.Image, or pg.Surface.')
        
        # Reset index to 0, but discard event because we'll manually trigger it later. This ensures event is triggered even if index is already 0.
        with param.discard_events(self):
            self.index = 0
        
        # Manually trigger index param to update texture
        self.param.trigger('index')
        
        return self
    
    def clear(self):
        """Clear image list. This will also clear the image window."""
        self._imgs.clear()
        with param.discard_events(self):
            self.index = 0
        self.param.trigger('index')
    
    @param.depends('index', watch=True)
    def _update_texture(self):
        """Update texture and render image defined by `index` param
        """
        if self._window is None:
            return

        GL = self._import_gl()

        try:
            win_w, win_h = pg.display.get_window_size()
            if (
                int(win_w) > 0
                and int(win_h) > 0
                and (int(win_w), int(win_h)) != self._window_size
            ):
                self._window_size = (int(win_w), int(win_h))
                self._setup_gl_state()
        except Exception:
            pass

        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        if len(self._imgs) > 0:
            img = self._imgs[self.index % len(self._imgs)]

            if isinstance(img, (str, pathlib.Path)):
                img = pg.image.load(img)

            if isinstance(img, pg.Surface):
                img = self._pad_surface_if_needed(img)
            else:
                raise TIPLMDisplayException(
                    'Error drawing image. Type must resolve to pygame Surface.'
                )

            tex_w, tex_h = img.get_size()
            tex_data = pg.image.tostring(img, 'RGB', True)

            GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._gl_tex_id))
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGB,
                int(tex_w),
                int(tex_h),
                0,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                tex_data,
            )

            # Match ImageWindow behavior: draw at top-left without interpolation.
            x = 0.0
            y = 0.0

            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2f(0.0, 1.0); GL.glVertex2f(x, y)
            GL.glTexCoord2f(1.0, 1.0); GL.glVertex2f(x + tex_w, y)
            GL.glTexCoord2f(1.0, 0.0); GL.glVertex2f(x + tex_w, y + tex_h)
            GL.glTexCoord2f(0.0, 0.0); GL.glVertex2f(x, y + tex_h)
            GL.glEnd()

        pg.display.flip()

    def next(self):
        """Display next image if multiple images have been loaded"""
        if len(self._imgs) > 0:
            self.index = (self.index + 1) % len(self._imgs)
        return self
    
    def previous(self):
        """Display previous image if multiple images have been loaded"""
        if len(self._imgs) > 0:
            self.index = (self.index - 1) % len(self._imgs)
        return self
    
    def on_keydown(self, key):
        """Handle keydown event to trigger next/previous image"""
        if key == pg.K_TAB or key == pg.K_RIGHT or key == pg.K_SPACE:
            self.next()
        elif key == pg.K_LEFT:
            self.previous()


