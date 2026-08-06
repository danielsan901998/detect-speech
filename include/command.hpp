#pragma once

#include <vector>
#include <string>

// Executes a command given an array of arguments without invoking a shell.
// Returns the exit code of the child process, or -1 on error.
int run_command(const std::vector<std::string>& args);
