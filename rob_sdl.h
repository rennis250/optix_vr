#pragma once

// #include <gl/GL.h>
#include "glad/glad.h"

#include "SDL.h"
#include "SDL_opengl.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdbool.h>

namespace rob {
	class SDLApp {
	public:
		SDLApp(unsigned int width, unsigned int height);
		~SDLApp();

		void registerInput(int &x, int& y);
		void clearScreen();
		void drawScene();

	private:
		SDL_Window* m_window = nullptr;
		SDL_Renderer* m_renderer = nullptr;
		SDL_GLContext m_gl_context = {};

		unsigned int m_width = 0;
		unsigned int m_height = 0;
	};

	SDLApp::SDLApp(unsigned int width, unsigned int height) {
		m_width = width;
		m_height = height;

		if (SDL_Init(SDL_INIT_VIDEO) < 0) {
			std::cerr << "shit sdl" << std::endl;
			exit(1);
		}

		// Default OpenGL is fine.
		SDL_GL_LoadLibrary(NULL);
		
		// Request an OpenGL 4.1 context (should be core)
		SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
		SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
		SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

		SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
		SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
		
		m_window = SDL_CreateWindow("SDL2 Test", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, m_width, m_height, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
		if (!m_window) {
			std::cerr << "shit window" << std::endl;
			exit(1);
		}

		m_gl_context = SDL_GL_CreateContext(m_window);

		m_renderer = SDL_CreateRenderer(m_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
		if (!m_renderer) {
			std::cerr << "shit renderer" << std::endl;
			exit(1);
		}

		SDL_GL_SetSwapInterval(-1);

		// Check OpenGL properties
		printf("OpenGL loaded\n");
		gladLoadGLLoader(SDL_GL_GetProcAddress);
		printf("Vendor:   %s\n", glGetString(GL_VENDOR));
		printf("Renderer: %s\n", glGetString(GL_RENDERER));
		printf("Version:  %s\n", glGetString(GL_VERSION));

		// Disable depth test and face culling.
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		glViewport(0, 0, m_width, m_height);
	}

	SDLApp::~SDLApp() {
		SDL_DestroyRenderer(m_renderer);
		SDL_DestroyWindow(m_window);
		SDL_Quit();
	}

	void SDLApp::registerInput(int &x, int &y) {
		SDL_Event event;
		Uint32 buttons;

		while (SDL_PollEvent(&event)) {
			switch (event.type) {
			case SDL_QUIT:
				exit(0);
				break;
			default:
				buttons = SDL_GetMouseState(&x, &y);
				break;
			}
		}

		return;
	}

	void SDLApp::clearScreen() {
		glClearColor(1.0f, 0.0f, 1.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	void SDLApp::drawScene() {
		SDL_GL_SwapWindow(m_window);
	}
}