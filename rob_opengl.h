#pragma once

#include "glad/glad.h"

#include <string>
#include <iostream>

namespace rob {
	const std::string s_vert_source = R"(
#version 410 core

layout(location = 0) in vec3 vertexPosition_modelspace;
out vec2 UV;

void main()
{
	gl_Position =  vec4(vertexPosition_modelspace, 1);
	UV = (vec2(vertexPosition_modelspace.x, vertexPosition_modelspace.y) + vec2(1, 1))/2.0;
}
)";

	const std::string s_frag_source = R"(
#version 410 core

in vec2 UV;
out vec4 color;

uniform sampler2D render_tex;
uniform bool correct_gamma;

void main()
{
    // color = texture(render_tex, UV).xyz;
	color = texture(render_tex, UV);
}
)";

	class GLState {
	public:
		void buildPBO(const unsigned int width, const unsigned int height);
		void buildVAO();
		void buildVBOandEBO();
		void buildShaderProgram();
		void createTexture();
		void draw();

		GLuint m_pbo;

	private:
		float m_vertices[12] = {
			 1.0f,  1.0f, 0.0f,  // top right
			 1.0f, -1.0f, 0.0f,  // bottom right
			-1.0f, -1.0f, 0.0f,  // bottom left
			-1.0f,  1.0f, 0.0f   // top left 
		};

		unsigned int m_indices[6] = {  // note that we start from 0!
			0, 1, 3,   // first triangle
			1, 2, 3    // second triangle
		};

		GLuint m_tex;

		unsigned int m_width = 0;
		unsigned int m_height = 0;
			 
		unsigned int m_VAO = 0;
		unsigned int m_VBO = 0;
		unsigned int m_EBO = 0;
		unsigned int m_vertexShader = 0;
		unsigned int m_fragmentShader = 0;
		unsigned int m_shaderProgram = 0;

		GLint m_tex_uniform_loc = -1;
	};

	void GLState::createTexture() {
		glGenTextures(1, &m_tex);
		glBindTexture(GL_TEXTURE_2D, m_tex);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	void GLState::buildPBO(const unsigned int width, const unsigned int height) {
		m_width = width;
		m_height = height;

		glGenBuffers(1, &m_pbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_pbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(uchar4) * (size_t)width * (size_t)height, nullptr, GL_STREAM_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0u);

		return;
	}

	void GLState::buildVBOandEBO() {
		glGenBuffers(1, &m_VBO);
		glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(m_vertices), m_vertices, GL_STATIC_DRAW);
		
		glGenBuffers(1, &m_EBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(m_indices), m_indices, GL_STATIC_DRAW);

		return;
	}

	void GLState::buildShaderProgram() {
		m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
		const GLchar* src = reinterpret_cast<const GLchar*>(s_vert_source.data());
		glShaderSource(m_vertexShader, 1, &src, NULL);
		glCompileShader(m_vertexShader);

		int  success;
		char infoLog[512];
		glGetShaderiv(m_vertexShader, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(m_vertexShader, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		m_fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		src = reinterpret_cast<const GLchar*>(s_frag_source.data());
		glShaderSource(m_fragmentShader, 1, &src, NULL);
		glCompileShader(m_fragmentShader);
		glGetShaderiv(m_fragmentShader, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(m_fragmentShader, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		m_shaderProgram = glCreateProgram();
		glAttachShader(m_shaderProgram, m_vertexShader);
		glAttachShader(m_shaderProgram, m_fragmentShader);
		glLinkProgram(m_shaderProgram);

		glGetProgramiv(m_shaderProgram, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(m_shaderProgram, 512, NULL, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		glUseProgram(m_shaderProgram);

		m_tex_uniform_loc = glGetUniformLocation(m_shaderProgram, "render_tex");

		glDeleteShader(m_vertexShader);
		glDeleteShader(m_fragmentShader);

		return;
	}

	void GLState::buildVAO() {
		glGenVertexArrays(1, &m_VAO);
		glBindVertexArray(m_VAO);

		return;
	}

	void GLState::draw() {
		// glUseProgram(m_shaderProgram);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_tex);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		glUniform1i(m_tex_uniform_loc, 0);

		glBindVertexArray(m_VAO);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glDisableVertexAttribArray(0);

		return;
	}
}