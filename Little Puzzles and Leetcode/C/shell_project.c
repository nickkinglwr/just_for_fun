// Nicholas Lower
// Stunted shell

#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <termios.h>

#define BUFF_SIZE 256
#define COMMAND_LIMIT 20

typedef struct histList histList;
struct histList { char *cmd; histList *next;  histList *prev; };
		    
void execCmds(char*);
char** storeCmds(char***, char*);
void mergeCmd(char*);
void addHist(char*, histList**, histList**);

int main()
{
  char c = 0;
  char *cwdbuf, *execbuf, *tokens;
  int top = 0, loc = 0, i = 0;
  histList *head, *tail, *curr;
  head = tail = curr = NULL;

  struct termios origConfig; // store old termios config
  tcgetattr(0, &origConfig);
  struct termios newConfig = origConfig;

  newConfig.c_lflag &= ~(ICANON|ECHO); // set new input/output flags
  newConfig.c_cc[VMIN] = 5;
  newConfig.c_cc[VTIME] = 2;

  tcsetattr(0, TCSANOW, &newConfig); // set new config

  while(1)
    {
      cwdbuf = (char*)malloc(BUFF_SIZE * sizeof(char));
      getcwd(cwdbuf, BUFF_SIZE); // put current directory in cwdbuf then output; is prompt
      printf("%s", cwdbuf);
      putchar('>');
      execbuf = (char*)malloc(BUFF_SIZE * sizeof(char));

      while(c != '\n') // continue to get/put input chars while not newline
	{
	  c = getchar();

	  if(c == 127 || c == 8) // if backspace or delete char then perform backspace steps
	    {
	      if(loc != 0) // only backspace if cursor location is not 0, saves prompt from backspace
		{
		  putchar('\b');
		  putchar(' ');
		  putchar('\b');
		  --loc;
		  --top;	      
		}
	      continue;
	    }

	  else if(c == 27)
	    {
	      c = getchar();
	      if(c == 91) // if first char was 27 and second was 91 check to see if third completes arrow key chain
		{
		  c = getchar();
		  if(c == 68) // if third char is 68 then left arrow was entered
		    {
		      if(loc != 0) // if cursor location is not at end of input put \b and dec cursor location
			{
			  putchar('\b');
			  --loc;
			}
		      continue;
		    }

		  else if(c == 67) // if third char is 67 then right arrow was entered
		    {
		      if(loc < top) // if cursor location is below the end of input then output current cursor char and inc location
			{			
			  putchar(execbuf[loc]);
			  ++loc;
			}
		      continue;
		    }

		  else if(c == 65) // if third char is 65 then up arrow was entered
		    {
		      if(curr == NULL) // only get history if curr points to history	      
				continue;

		      if(curr == head && top != 0)
			  addHist(execbuf, &head, &tail);

		      loc = top;
		      for(i=top; i > 0; --i) // remove all current input chars
			{
			  putchar('\b');
			  putchar(' ');
			  putchar('\b');
			}
		      loc = 0;
		      top = 0;
		      c = curr->cmd[top];
		      while(c != 0) // while c is not 0 put the stored histArr cmd chars to input and exec buffer while saving new input size and cursor loc
			{
			  putchar(c);
			  execbuf[top] = c;
			  ++top;
			  ++loc;
			  c = curr->cmd[top];
			}
		      if(curr != tail) // only set curr to next history if it is not the tail
			curr = curr->next;
		      
		      continue;
		    }

		  else if(c == 66) 
		    {
		      if(curr == NULL || curr->prev == NULL) // if curr is NULL continue without resetting input
			continue;

		      curr = curr->prev;

		      loc = top;
		      for(i=top; i > 0; --i) // remove all current input chars
			{
			  putchar('\b');
			  putchar(' ');
			  putchar('\b');
			}
		      loc = 0;
		      top = 0;
		      c = curr->cmd[top];
		      while(c != 0) // while c is not 0 put the stored histArr cmd chars to input and exec buffer while saving new input size and cursor loc
			{
			  putchar(c);
			  execbuf[top] = c;
			  ++top;
			  ++loc;
			  c = curr->cmd[top];
			}
		      continue;
		    }
		}
	    }	

	  putchar(c); // print current char
	  execbuf[top] = c; // save char to execute buffer at current top location
	  ++top; // inc input count and cursor location
	  ++loc;
	}
      execbuf[top] = 0;
      addHist(execbuf, &head, &tail);
      curr = head;
      tokens = strtok(execbuf, " \n");
      top = 0; //reset input variables
      loc = 0;
      c = 0;

      if(tokens == NULL)
	continue;

      if(strcmp("quit", tokens) == 0) // if imput is quit then display quit prompt and getchar, if y then execute main while
	{
	  free(cwdbuf);
	  free(execbuf);
	  printf("Are you sure you want to quit? (y/n)\n");
	  c = getchar(); 
	  if(c == 'y')
	    break;
	  else
	    continue;
	}
      else if(strcmp("merge",tokens)==0) // if input is merge call merge command function with tokens
	mergeCmd(tokens);

      else if(strcmp("pause", tokens) == 0) // if input is pause then pause execution by calling while loop until enter is pressed
	while(getchar() != '\n'){}

      else if(strcmp("cd",tokens)==0) // if input is cd call chdir on new path
	{
	  if(chdir(strtok(NULL, " \n")) < 0)
	    perror(NULL);
	}
      else if(strcmp("dir", tokens) == 0) // if input is dir call execCmds with ls instead
	execCmds("ls");

      else
	execCmds(tokens); // else all other if/elseifs fail then it is a command to execute, so call execCmds on tokens

      free(cwdbuf);
      free(execbuf);
    }
  curr = head;
  while(curr != NULL) // free hist list memory
    {
      free(curr->cmd);
      free(curr);
      curr = curr->next;
    }

  tcsetattr(0, TCSANOW, &origConfig); //return original input config
  return 1;
}


void execCmds(char * tokens)
{
  int status;
  int fd[2];

  char *** commands = (char***)malloc(2 * sizeof(char**)); // commands ponits to array of each set of commands, set to 2 to only handle 2 commands for one pipe
  commands[1] = NULL; // NULL signals that no pipe is present
  commands[0] = storeCmds(commands, tokens); // set commands

  pid_t pid = fork();
       		   
  if(pid > 0)
    waitpid(pid, &status, WUNTRACED); // if parent then wait for child to terminate
  
  else if(pid == 0)	{
    if(commands[1] != NULL) { // if there is a pipe, then execute pipe
      pipe(fd);
      pid_t pid2 = fork(); // fork another child from child in order to execute both commands without exiting main process

      if(pid2 > 0) {
	close(0);
	dup(fd[0]); // make this process get input from pipe
	close(fd[1]);

	waitpid(pid2, &status, WUNTRACED);
	execvp(commands[1][0], commands[1]);
	perror(NULL); // if execvp did not terminate process then error occur, output error then manually exit
	exit(0);
      }
      else if(pid2 == 0){
	close(1);
	dup(fd[1]); // make this process output to pipe
	close(fd[0]);

	execvp(commands[0][0], commands[0]);
	perror(NULL); // if execvp did not terminate process then error occur, output error then manually exit
	exit(0);
      }
    }
    else { // else if there is no pipe then excute single commmand
      execvp(commands[0][0], commands[0]);
      perror(NULL); // if execvp did not terminate process then error occur, output error then manually exit
      exit(0);
    }
  }
  else
    perror(NULL);

  int i=0;
  while(commands[0][i] != NULL)
    {
      free(commands[0][i]);
      ++i;
    }
  i=0;
  if(commands[1] != NULL){
    while(commands[1][i] != NULL)
      {
	free(commands[1][i]);
	++i;
      }}
  free(commands[0]);
  free(commands[1]);
  free(commands);
}


char** storeCmds(char*** commands, char* currToken ) 
{
  if(currToken == NULL) // if there are no tokens then return NULL
    return NULL;

  char ** cmds = (char**)malloc(COMMAND_LIMIT * sizeof(char*)); // initiliaze this set of commands to COMMAND_LIMIT
  int i=0;
  while(currToken != NULL) // while there are tokens
    {
      if(*currToken == '|') // if the token is a pipe then reallocate current set then recursivaly call function on remaining commands
	{
	  cmds = (char**)realloc(cmds, (i+1)*sizeof(char*));
	  cmds[i] = NULL;
	  currToken = strtok(NULL, " \n");
	  commands[1] = storeCmds(commands, currToken);
	  return cmds;
	} 
      cmds[i] = (char*)malloc(strlen(currToken) * sizeof(char)); // allocate current cell to size of token
      strcpy(cmds[i], currToken); // copy token into memory
      currToken = strtok(NULL, " \n"); // get next token
      ++i;
    }
  cmds = (char**)realloc(cmds, (i+1)*sizeof(char*)); // reallocate commands
  cmds[i] = NULL;
  return cmds; // return pointer to set of commands
}


void mergeCmd(char* tokens)
{
  char ** cmd = (char **)malloc(4 * sizeof(char*)); // make cmd an array of strings to hold cat and first two files
  FILE * outFile;
  cmd[0] = "cat"; //set cmd to cat
  tokens = strtok(NULL, " \n");
  int i = 1, pid, status;

  while(*tokens != '>')
    {
      cmd[i] = (char*)malloc(strlen(tokens) * sizeof(char));
      strcpy(cmd[i], tokens); // copy cat files into cmd array
      tokens = strtok(NULL, " \n");
      ++i;
    }
  cmd[i] = NULL;
  
  pid = fork(); // fork new process

  if(pid > 0)
    waitpid(pid, &status,WUNTRACED);
  else if(pid == 0)
    {
      tokens = strtok(NULL, " \n");
      outFile = fopen(tokens, "w"); //open third file
      if(outFile == NULL)
	{
	  perror(NULL);
	  exit(0);
	}

      close(1);
      dup(fileno(outFile)); // set stdout of this child process to third file and exec cat on first 2 files

      execvp(cmd[0], cmd);
      perror(NULL);
      exit(0);
    }
  else
    perror(NULL);
}


void addHist(char* cmd, histList **head, histList **tail)
{
  histList* tmp = (histList*)malloc(sizeof(histList)); // tmp to hold new cmd to be inserted
  tmp->cmd = (char*)malloc(strlen(cmd) * sizeof(char));
  strcpy(tmp->cmd, cmd); // move command into tmp's cmd field
  tmp->cmd[strlen(cmd)-1] = 0; // replace \n with 0
  tmp->prev = NULL;
  tmp->next = *head;

  if(*head == NULL && *tail == NULL) // if there is no element in list set tail and head to tmp
    {
      *head = tmp;
      *tail = tmp; 
    }
  else // else set new head
    {
      (*head)->prev = tmp;
      *head = tmp;
    }
}
