#include <vector>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

// Executes a command given an array of arguments without invoking a shell.
// Returns the exit code of the child process, or -1 on error.
int run_command(const std::vector<std::string>& args) {
    if (args.empty()) {
        return -1;
    }

    pid_t pid = fork();

    if (pid < 0) {
        // Fork failed
        return -1;
    } 
    
    if (pid == 0) {
        // Child process: build NULL-terminated array of char pointers
        std::vector<char*> c_args;
        c_args.reserve(args.size() + 1);
        for (const auto& arg : args) {
            c_args.push_back(const_cast<char*>(arg.c_str()));
        }
        c_args.push_back(nullptr);

        execvp(c_args[0], c_args.data());
        // Reached only if execvp fails
        _exit(127);
    }

    // Parent process: wait for child to finish
    int status = 0;
    if (waitpid(pid, &status, 0) == -1) {
        return -1;
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }

    return -1;
}
