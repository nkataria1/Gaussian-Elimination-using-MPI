/* Gaussian elimination without pivoting.
   irintf(" ------------------------------------PRINTING AFETER NORM %d----------------------------\n", norm);
   print_inputs();
 * Compile with "gcc gauss.c" 
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <mpi.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */

/*--------------------------PARAMETERS INTRODUCED AS PART OF MPI----------------------*/
int N;  /* Matrix size */
int ierr; /* Error code for MPI Messages */
int num_procs; /* Total Number of Processes */
MPI_Status status; /* Status for MPI_Recv */
int my_rank; /* MPI Rank for the current process */

/* Function to distribute rows to the processes */
void parallelize_row(int norm);

/* Function to collate computed rows from all the processes */
void collate_rows(norm);

/* Function to broadcast norm to all processes with rank more than my_rank */ 
void send_norm_to_all_processes (float arrayA[], float arrayB, int my_rank);

/* Function to calculate multiplier for the current row */
float calculate_multiplier (int norm, float A[], float B, float AN[], float normB);

#define GAUSS_SEND_TAG 2001
#define GAUSS_RECV_TAG 2002
/*------------------------------------------------------------------------------------*/

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3

/* Prototype */
void gauss();  /* The function you will provide.
		* It is this routine that is timed.
		* It is called only on the parent.
		*/

/* returns a seed for srand based on the time */
unsigned int time_seed() {
	struct timeval t;
	struct timezone tzdummy;

	gettimeofday(&t, &tzdummy);
	return (unsigned int)(t.tv_usec);
}

/* Set the program parameters from the command-line arguments */
void parameters(int argc, char **argv) {
	int seed = 0;  /* Random seed */
	char uid[32]; /*User name */

	/* Read command-line arguments */
	srand(time_seed());  /* Randomize */

	if (argc == 3) {
		seed = atoi(argv[2]);
		srand(seed);
		printf("Random seed = %i\n", seed);
	} 


	if (argc >= 2) {
		N = atoi(argv[1]);
		if (N < 1 || N > MAXN) {
			printf("N = %i is out of range.\n", N);
			exit(0);
		}
	}
	else {
		printf("Usage: %s <matrix_dimension> [random seed]\n",
				argv[0]);    
		exit(0);
	} 

	// Print parameters 
	printf("\nMatrix dimension N = %i.\n", N);

}

/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
	int row, col;

	printf("\nInitializing...\n");
	for (col = 0; col < N; col++) {
		for (row = 0; row < N; row++) {
			A[row][col] = (float)rand() / 32768.0;
		}
		B[col] = (float)rand() / 32768.0;
		X[col] = 0.0;
	}

}

/* Print input matrices */
void print_inputs() {
	int row, col;

	if (N < 10) {
		printf("\nA =\n\t");
		for (row = 0; row < N; row++) {
			for (col = 0; col < N; col++) {
				printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
			}
		}
		printf("\nB = [");
		for (col = 0; col < N; col++) {
			printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
		}
	}
}

void print_X() {
	int row;

	if (N < 500) {
		printf("\nX = [");
		for (row = 0; row < N; row++) {
			printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
		}
	}
}

int main(int argc, char **argv) {
	/* Timing variables */
	struct timeval etstart, etstop;  /* Elapsed times using gettimeofday() */
	struct timezone tzdummy;
	clock_t etstart2, etstop2;  /* Elapsed times using times() */
	unsigned long long usecstart, usecstop;
	struct tms cputstart, cputstop;  /* CPU times for my processes */

	ierr = MPI_Init(&argc, &argv);
	ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	/* create a type for struct car */

	if (my_rank == 0)  { // root process
		/* Process program parameters */
		parameters(argc, argv);

		/* Initialize A and B */
		initialize_inputs();

		/* Print input matrices */
		print_inputs();
	}
	/* start clock */
	if (my_rank == 0)  { // root process
		printf("\nstarting clock. %d\n", my_rank);
		gettimeofday(&etstart, &tzdummy);
		etstart2 = times(&cputstart);
	}
	
	/* Broadcast N to all processes */
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

	/* Gaussian Elimination */
	gauss(my_rank);

	/* Stop Clock */
	if (my_rank == 0)  { // root process
		gettimeofday(&etstop, &tzdummy);
		etstop2 = times(&cputstop);
		printf("Stopped clock.\n");
		usecstart = (unsigned long long)etstart.tv_sec * 1000000 + etstart.tv_usec;
		usecstop = (unsigned long long)etstop.tv_sec * 1000000 + etstop.tv_usec;

		/* Display output */
		print_X();

		/* Display timing results */
		printf("\nElapsed time = %g ms.\n",
				(float)(usecstop - usecstart)/(float)1000);

		printf("(CPU times are accurate to the nearest %g ms)\n",
				1.0/(float)CLOCKS_PER_SEC * 1000.0);
		printf("My total CPU time for parent = %g ms.\n",
				(float)( (cputstop.tms_utime + cputstop.tms_stime) -
					(cputstart.tms_utime + cputstart.tms_stime) ) /
				(float)CLOCKS_PER_SEC * 1000);
		printf("My system CPU time for parent = %g ms.\n",
				(float)(cputstop.tms_stime - cputstart.tms_stime) /
				(float)CLOCKS_PER_SEC * 1000);
		printf("My total CPU time for child processes = %g ms.\n",
				(float)( (cputstop.tms_cutime + cputstop.tms_cstime) -
					(cputstart.tms_cutime + cputstart.tms_cstime) ) /
				(float)CLOCKS_PER_SEC * 1000);
		/* Contrary to the man pages, this appears not to include the parent */
		printf("--------------------------------------------\n");
	}
	exit(0);
}

/* Function to collate computed rows from all the processes */
void collate_rows(norm) {

	int total_num_rows, rows_per_process, remainder = 0, recv, rank, num_rows_to_recv, start_row,i;

	/* Calculate number of rows to receive data for */
	total_num_rows = N-1; 

	/* Calculate how many rows each process will send */
	rows_per_process = total_num_rows/num_procs;

	if (total_num_rows > (num_procs * rows_per_process)) {
		/* Check if number of processes is a perfect factor */
		remainder = (total_num_rows) % (num_procs);
	}

	/* Received so far ?? */
	recv = rows_per_process + norm + 1;

	for (rank = 1; rank < num_procs; rank++)  {

		/* Calculate how much data needs to be received 
 		 * from process with rank specified by variable 'rank'*/ 
		if (remainder > 0 ) {
			num_rows_to_recv = rows_per_process + 1;
			remainder--; 
		} else {
			num_rows_to_recv = rows_per_process;
		}

		if (num_rows_to_recv > (N - recv)) {
			num_rows_to_recv == N - recv;
		}

		start_row = recv; 

		/* Receive computed data from all processses */
		for (i=start_row ; i< start_row + num_rows_to_recv; i++) {	
			MPI_Recv( &A[i], N, MPI_FLOAT, rank,
					GAUSS_RECV_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv( &B[i], 1, MPI_FLOAT, rank,
					GAUSS_RECV_TAG, MPI_COMM_WORLD, &status);
		}
		recv = recv + num_rows_to_recv; 
	}
} /* end collate_rows */

/* Function to distribute rows to the processes */
void parallelize_row(int norm) {

	int total_num_rows, rows_per_process, remainder = 0, sent, rank, num_rows_to_send, start_row,i;

	/* Calculate number of rows to distribute */
	total_num_rows = N-1; 

	/* Calculate how many rows each process will get */
	rows_per_process = total_num_rows/num_procs;

	if (total_num_rows > (num_procs * rows_per_process)) {
		/* Check if number of processes is a perfect factor */
		remainder = (total_num_rows) % (num_procs);
	}

	/* Distributed so far ?? */
	sent = norm + 1 + rows_per_process;

	/* Calculate root's part of rows before distribution
           as other processes depend on it */
	int row, col;
	float multiplier;
	int inorm = norm;
	for (inorm = 0; inorm < sent; inorm ++) {
		for (row=inorm+1 ; row < sent; row++) {
			multiplier = A[row][inorm] / A[inorm][inorm];
			for (col = inorm; col < N; col++) {
				A[row][col] -= A[inorm][col] * multiplier;
			}
			B[row] -= B[inorm] * multiplier;
		}
	}


	/* Ok. Let's distribute :) */
	for (rank = 1; rank < num_procs; rank++)  {

		/* Calculate how many rows to send to the processes
  		 * with rank specified by 'rank' variable */
		if (remainder > 0 ) {
			num_rows_to_send = rows_per_process + 1;
			remainder--; 
		} else {
			num_rows_to_send = rows_per_process;
		}


		if (num_rows_to_send > (N- sent)) {
			num_rows_to_send == N - sent;
		}

		start_row = sent;

		/* Send metadata about the rows the process is responsible for */
		ierr = MPI_Send(&start_row, 1, MPI_INT, rank, GAUSS_SEND_TAG, MPI_COMM_WORLD);
		ierr = MPI_Send(&num_rows_to_send, 1, MPI_INT, rank, GAUSS_SEND_TAG, MPI_COMM_WORLD);

		/* Send the actual data to the processes */
		for (i=start_row; i<start_row+num_rows_to_send; i++) {
			ierr = MPI_Send(&A[i], N, MPI_FLOAT,
					rank, GAUSS_SEND_TAG, MPI_COMM_WORLD);
			ierr = MPI_Send(&B[i], 1, MPI_FLOAT,
					rank, GAUSS_SEND_TAG, MPI_COMM_WORLD);
		}
		sent = sent + num_rows_to_send;

	}

	/* Now that distribution is done, broadcast my calculations to all processes */
	for (row=norm; row <= norm+rows_per_process; row++) {
		send_norm_to_all_processes (A[row], B[row], 0);
	}
} /* end parallelize_rows */

/* Function to broadcast norm to all processes with rank more than my_rank */ 
void send_norm_to_all_processes (float arrayA[], float arrayB, int my_rank) {
	int i = 0, j;
	for (i=my_rank+1; i<num_procs; i++) {
		MPI_Send(&arrayA[0], N, MPI_FLOAT, i, GAUSS_SEND_TAG, MPI_COMM_WORLD);
		MPI_Send(&arrayB, 1, MPI_FLOAT, i, GAUSS_SEND_TAG, MPI_COMM_WORLD);
	}
} /* end send_norm_to_all_processes */

/* Function to calculate multiplier for the current row */
float calculate_multiplier (int norm, float A[], float B, float AN[], float normB) {
	int col;
	float multiplier;
	multiplier = A[norm] / AN[norm];
	for (col = norm; col < N; col++) {
		A[col] -= AN[col] * multiplier;
	}
	return (B - normB * multiplier);
} /* end calculate multiplier */

/* ------------------ Above Was Provided --------------------- */

/****** You will replace this routine with your own parallel version *******/
/* Provided global variables are MAXN, N, A[][], B[], and X[],
 * defined in the beginning of this code.  X[] is initialized to zeros.
 */
void gauss(int rank) {
	int norm, row, col;  /* Normalization row, and zeroing
			      * element row and col */
	if (rank == 0) {
		/* Gaussian elimination */

		/* Parallelize the inner loops */
		parallelize_row(0);

		/* Done distributing.. now collate the results */
		collate_rows(0);

		/* (Diagonal elements are not normalized to 1.  This is treated in back
		 * substitution.)
		 */

		/* Back substitution */
		for (row = N - 1; row >= 0; row--) {
			X[row] = B[row];
			for (col = N-1; col > row; col--) {
				X[row] -= A[row][col] * X[col];
			}
			X[row] /= A[row][row];
		}
	} else {
		int num_rows_to_receive, i, j, start_row;
		float NORMB;

		/* Receive the metadata about the rows this process is going to handle */

		/* Row number to start from */
		ierr = MPI_Recv(&start_row, 1, MPI_INT, 0, GAUSS_SEND_TAG, MPI_COMM_WORLD, &status);

		/* Number of rows from start_row that this process needs to compute */
		ierr = MPI_Recv(&num_rows_to_receive, 1, MPI_INT, 0, GAUSS_SEND_TAG, MPI_COMM_WORLD, &status);
		float arrayN[N], arrayA[num_rows_to_receive][N], arrayB[num_rows_to_receive];
		/* -------------------------Done Receiving Metadata-------------------------------*/

		/* Receive the actual row data now */
		for (i = 0; i < num_rows_to_receive; i++) {
			ierr = MPI_Recv( &arrayA[i], N, MPI_FLOAT,
					0, GAUSS_SEND_TAG, MPI_COMM_WORLD, &status);
			ierr = MPI_Recv( &arrayB[i], N, MPI_FLOAT,
					0, GAUSS_SEND_TAG, MPI_COMM_WORLD, &status);
		}
		/*-------------------------Done Receiving the Rows Data--------------------------*/


		/* Start computations..................................................*/
		for (norm = 0 ; norm < start_row + num_rows_to_receive; norm++) {
			/* Check if the current process has the prequired norm */
			if ((start_row <= norm)  && (norm < num_rows_to_receive+start_row)) {
				for (i=0; i<num_rows_to_receive; i++) {
					if (i > (norm-start_row)) {
						arrayB[i] = calculate_multiplier(norm, &arrayA[i], arrayB[i], &arrayA[norm-start_row], arrayB[norm-start_row]); 
					}
				}
			} else  { /* We need to receive norm from another process */
				ierr = MPI_Recv(&arrayN, N, MPI_FLOAT, MPI_ANY_SOURCE, GAUSS_SEND_TAG, MPI_COMM_WORLD, &status);
				ierr = MPI_Recv(&NORMB, 1, MPI_FLOAT, MPI_ANY_SOURCE, GAUSS_SEND_TAG, MPI_COMM_WORLD, &status);
				/*-------------------------Done receiving the norm data--------------------------------------*/

				/* For each row, compute multiplier */	
				for (i=0; i<num_rows_to_receive; i++) {
					if (i > (norm-start_row)) {
						arrayB[i]  = calculate_multiplier(norm, &arrayA[i], arrayB[i], &arrayN, NORMB); 
					}
				}
				/*-------------------------Done computing the multiplier--------------------------------------*/
			}

			/* Broadcast my norm's data to processes more than my rank */
			if (my_rank != num_procs - 1) { /* Last process ?? No need to broadcast */
				if ((start_row <=norm)  && (norm <= num_rows_to_receive+start_row)) {
					/* Its this processes's turn to send computed norm row to 
  					 * Processes with rank more that its rank */
					send_norm_to_all_processes(arrayA[norm-start_row], arrayB[norm-start_row], my_rank); 
				}
			}
		}

		/* Done Computing the rows, Send it to the root process for collation */
		for (i = 0; i < num_rows_to_receive; i++) {
			ierr = MPI_Send( &arrayA[i], N, MPI_FLOAT, 0, GAUSS_RECV_TAG, MPI_COMM_WORLD);
			ierr = MPI_Send( &arrayB[i],1, MPI_FLOAT, 0, GAUSS_RECV_TAG, MPI_COMM_WORLD);
		}
		/*------------------------------Done !! --------------------------------------------*/
	}
	/* Finalize the process for graceful termination */
	MPI_Finalize();
}
