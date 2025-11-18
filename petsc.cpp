static char help[] = "Solves a linear system from CSR input using PCG.\n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
    Mat             A;          /* Linear system matrix */
    Vec             x, b, u;    /* approx solution, RHS, exact solution */
    KSP             ksp;        /* linear solver context */
    PC              pc;         /* preconditioner context */
    PetscErrorCode  ierr;
    PetscMPIInt     rank, size;
    char            file[PETSC_MAX_PATH_LEN]; /* Input file name */
    PetscBool       flg;
    PetscViewer     view;

    /* Initialize PETSc 
       This handles MPI initialization automatically.
    */
    PetscInitialize(&argc, &args, (char *)0, help);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    /* ----------------------------------------------------------------
       1. Load Matrix from File
       Requires command line argument: -f <filename>
    ---------------------------------------------------------------- */
    
    PetscOptionsGetString(NULL, NULL, "-f", file, PETSC_MAX_PATH_LEN, &flg);
    if (!flg) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Must indicate binary file with the -f option");

    /* Open the file and load the matrix */
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &view);
    MatCreate(PETSC_COMM_WORLD, &A);
    MatSetFromOptions(A);
    MatLoad(A, view);
    PetscViewerDestroy(&view);

    /* ----------------------------------------------------------------
       2. Setup Vectors
    ---------------------------------------------------------------- */
    /* MatCreateVecs automatically handles parallel distribution based on A */
    MatCreateVecs(A, &x, &b);
    VecDuplicate(x, &u); 

    /* Set exact solution u = [1, 1, ..., 1] */
    VecSet(u, 1.0);

    /* Compute RHS b = A * u */
    MatMult(A, u, b);

    /* ----------------------------------------------------------------
       4. Setup Solver (KSP)
       Here we explicitly ask for PCG (Preconditioned Conjugate Gradient)
    ---------------------------------------------------------------- */
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    
    /* Set operators: A * x = b */
    KSPSetOperators(ksp, A, A);

    /* HARDCODED CONFIGURATION: 
       Set Solver to Conjugate Gradient (KSPCG)
       Set Preconditioner to Jacobi (PCJACOBI) - standard diagonal scaling
    */
    KSPSetType(ksp, KSPCG);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCJACOBI);

    /* Allow user overrides via command line 
       (e.g., user can run with -pc_type bjacobi without recompiling)
    */
    KSPSetFromOptions(ksp);

    /* ----------------------------------------------------------------
       5. Solve
    ---------------------------------------------------------------- */
    KSPSolve(ksp, b, x);

    /* ----------------------------------------------------------------
       6. Check Convergence and Cleanup
    ---------------------------------------------------------------- */
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    
    if (rank == 0) {
        if (reason > 0) {
            PetscInt its;
            KSPGetIterationNumber(ksp, &its);
            PetscPrintf(PETSC_COMM_SELF, "Converged in %d iterations.\n", its);
        } else {
            PetscPrintf(PETSC_COMM_SELF, "Diverged! Reason: %d\n", reason);
        }
    }

    /* View the solution (only for small systems!) */
    PetscPrintf(PETSC_COMM_WORLD, "Solution Vector:\n");
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    /* Free work space */
    KSPDestroy(&ksp);
    VecDestroy(&u);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);

    PetscFinalize();
    return 0;
}