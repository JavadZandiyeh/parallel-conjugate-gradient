#include <cmath>
#include <iostream>
#include <map>
#include <mpi.h>
#include <vector>

// --- ASSUMED DATA STRUCTURES ---
// Based on common parallel library conventions for 1D row decomposition
struct CSR_local {
  int local_n;     // Number of rows this process owns
  int global_n;    // Total rows in global matrix
  int global_cols; // Total cols in global matrix
  long long nnz;   // Number of non-zeros this process owns

  // MPI info
  MPI_Comm comm;
  int mpi_rank;
  int mpi_size;

  // Row distribution info
  int row_start; // Global index of the first row this process owns
  int row_end;   // Global index of the row AFTER the last one (row_start +
                 // local_n)

  // CSR data for local rows
  std::vector<int> row_ptr;   // Size: local_n + 1
  std::vector<int> col_ind;   // Global column indices
  std::vector<double> values; // Non-zero values
};

// --- HELPER FUNCTIONS ---

/**
 * @brief Computes the parallel sparse matrix-vector product y = A*x.
 * This is the most complex part, requiring communication for off-process
 * elements.
 */
void parallel_spmv(const CSR_local &A, const std::vector<double> &x_local,
                   std::vector<double> &y_local) {
  // 1. Identify required non-local 'x' entries (the "halo")
  std::map<int, std::vector<int>> send_requests; // Key: rank to send to
  std::map<int, int> halo_indices; // Global index -> local index in recv_buf
  std::vector<double> halo_values; // Buffer to hold received x-values

  for (int i = 0; i < A.local_n; ++i) {
    for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
      int j_global = A.col_ind[k];

      // Check if the column index is off-process
      if (j_global < A.row_start || j_global >= A.row_end) {
        // Figure out which rank owns this column
        // (This assumes a simple, contiguous block distribution)
        int owner_rank = j_global / (A.global_n / A.mpi_size);
        // Handle last rank possibly having more rows
        if (owner_rank >= A.mpi_size)
          owner_rank = A.mpi_size - 1;

        if (send_requests.find(owner_rank) == send_requests.end()) {
          send_requests[owner_rank] = std::vector<int>();
        }
        send_requests[owner_rank].push_back(j_global);

        // Track this halo index if not already seen
        if (halo_indices.find(j_global) == halo_indices.end()) {
          halo_indices[j_global] = halo_values.size();
          halo_values.push_back(0.0); // Placeholder
        }
      }
    }
  }

  // 2. Perform All-to-all communication to exchange halo data
  // This is a simplified MPI_Alltoallv pattern.
  // A more robust implementation would use non-blocking I/O.
  std::vector<double> send_buf;
  std::vector<int> send_counts(A.mpi_size, 0);
  std::vector<int> recv_counts(A.mpi_size, 0);
  std::vector<int> send_displs(A.mpi_size, 0);
  std::vector<int> recv_displs(A.mpi_size, 0);

  // Prepare send buffers (what others requested from me)
  // This is the "reverse" of the send_requests map.
  // We need to know who needs *my* local x-values.
  // This is complex. A simpler (but less efficient) way is to use MPI_Allgather
  // to find out who needs what.

  // --- Simplified Communication (less efficient, easier to show) ---
  // A full, efficient implementation requires a 2-stage Alltoallv to
  // exchange request sizes, then the data itself.
  // For this example, we'll use a placeholder.
  // A *truly* robust SpMV would pre-calculate this communication pattern.

  // Let's assume a pre-computed communication pattern exists for simplicity.
  // For this example, we'll just focus on the SpMV logic.
  // In a real scenario, the setup of this communication is non-trivial.

  // 2. (Re-simplified) Multiply local and remote parts
  std::fill(y_local.begin(), y_local.end(), 0.0);

  // We'd need to fill 'halo_values' via MPI here.
  // Without that, this code will only work for diagonal matrices.
  // Let's assume 'x_local' magically contains all needed values
  // (local + halo) for this example to be self-contained.
  // A better approach: gather *all* of x.

  std::vector<double> x_global(A.global_n);
  std::vector<int> recvcounts(A.mpi_size);
  std::vector<int> displs(A.mpi_size);

  int local_size = A.local_n;
  MPI_Allgather(&local_size, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, A.comm);

  displs[0] = 0;
  for (int i = 1; i < A.mpi_size; ++i) {
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }

  // Gather all parts of vector x onto every process
  MPI_Allgatherv(x_local.data(), A.local_n, MPI_DOUBLE, x_global.data(),
                 recvcounts.data(), displs.data(), MPI_DOUBLE, A.comm);

  // 3. Now compute y = A*x using the global x vector
  for (int i = 0; i < A.local_n; ++i) {
    for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
      y_local[i] += A.values[k] * x_global[A.col_ind[k]];
    }
  }

  // NOTE: MPI_Allgatherv is NOT scalable. A true parallel SpMV uses
  // a "halo exchange" (MPI_Isend/Irecv or MPI_Alltoallv) with a
  // pre-computed communication pattern. But this will work for this example.
}

/**
 * @brief Computes the parallel dot product: d = x^T * y
 */
double parallel_dot(int local_n, const std::vector<double> &x_local,
                    const std::vector<double> &y_local, MPI_Comm comm) {
  double local_dot = 0.0;
  for (int i = 0; i < local_n; ++i) {
    local_dot += x_local[i] * y_local[i];
  }

  double global_dot = 0.0;
  MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, comm);
  return global_dot;
}

/**
 * @brief Computes the parallel axpy operation: y = a*x + y
 */
void parallel_axpy(int local_n, double a, const std::vector<double> &x_local,
                   std::vector<double> &y_local) {
  for (int i = 0; i < local_n; ++i) {
    y_local[i] = a * x_local[i] + y_local[i];
  }
}

/**
 * @brief Extracts the diagonal of the local matrix block for the Jacobi
 * preconditioner.
 */
void extract_local_diagonal(const CSR_local &A,
                            std::vector<double> &diag_local) {
  diag_local.assign(A.local_n, 0.0);
  for (int i = 0; i < A.local_n; ++i) {
    int global_row = A.row_start + i;
    for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
      if (A.col_ind[k] == global_row) {
        diag_local[i] = A.values[k];
        break;
      }
    }
    // Handle case where diagonal might be zero
    if (std::abs(diag_local[i]) < 1e-12) {
      diag_local[i] = 1.0;
    }
  }
}

/**
 * @brief Applies the Jacobi preconditioner: z = M_inv * r
 */
void apply_preconditioner(int local_n, const std::vector<double> &diag_local,
                          const std::vector<double> &r_local,
                          std::vector<double> &z_local) {
  for (int i = 0; i < local_n; ++i) {
    z_local[i] = r_local[i] / diag_local[i];
  }
}

/**
 * @brief Solves Ax = b using the parallel Preconditioned Conjugate Gradient
 * method.
 *
 * @param A The distributed CSR_local matrix.
 * @param b_local The local portion of the right-hand side vector 'b'.
 * @param x_local The local portion of the solution vector 'x' (initial guess,
 * modified in-place).
 * @param max_iter Maximum number of iterations.
 * @param tolerance Convergence tolerance.
 */
void parallel_pcg(const CSR_local &A, const std::vector<double> &b_local,
                  std::vector<double> &x_local, int max_iter,
                  double tolerance) {

  int rank = A.mpi_rank;
  MPI_Comm comm = A.comm;
  int n_local = A.local_n;

  // Allocate local vectors
  std::vector<double> r_local(n_local);  // Residual: r = b - A*x
  std::vector<double> p_local(n_local);  // Search direction
  std::vector<double> z_local(n_local);  // Preconditioned residual
  std::vector<double> Ap_local(n_local); // A * p

  // Preconditioner setup (Jacobi)
  std::vector<double> diag_local;
  extract_local_diagonal(A, diag_local);

  // --- Initialization ---
  // r = b - A*x (Compute initial residual)
  parallel_spmv(A, x_local, r_local); // r_local = A*x
  for (int i = 0; i < n_local; ++i) {
    r_local[i] = b_local[i] - r_local[i]; // r_local = b - A*x
  }

  // z = M_inv * r
  apply_preconditioner(n_local, diag_local, r_local, z_local);

  // p = z
  p_local = z_local;

  // rs_old = r^T * z
  double rs_old = parallel_dot(n_local, r_local, z_local, comm);
  double rs_init = rs_old; // For relative residual check

  if (rank == 0) {
    std::cout << "PCG Starting. Initial residual norm: " << std::sqrt(rs_init)
              << std::endl;
  }

  for (int iter = 0; iter < max_iter; ++iter) {
    // 1. Ap = A * p
    parallel_spmv(A, p_local, Ap_local);

    // 2. alpha = rs_old / (p^T * Ap)
    double p_dot_Ap = parallel_dot(n_local, p_local, Ap_local, comm);
    double alpha = rs_old / p_dot_Ap;

    // 3. x = x + alpha * p
    parallel_axpy(n_local, alpha, p_local, x_local);

    // 4. r = r - alpha * Ap
    parallel_axpy(n_local, -alpha, Ap_local, r_local);

    // --- Check for convergence ---
    double r_norm_sq = parallel_dot(n_local, r_local, r_local, comm);
    if (rank == 0) {
      std::cout << "  Iter " << iter
                << ": Rel. Residual Norm = " << std::sqrt(r_norm_sq / rs_init)
                << std::endl;
    }
    if (std::sqrt(r_norm_sq / rs_init) < tolerance) {
      if (rank == 0) {
        std::cout << "PCG Converged in " << iter + 1 << " iterations."
                  << std::endl;
      }
      break;
    }

    // 5. z = M_inv * r
    apply_preconditioner(n_local, diag_local, r_local, z_local);

    // 6. rs_new = r^T * z
    double rs_new = parallel_dot(n_local, r_local, z_local, comm);

    // 7. beta = rs_new / rs_old
    double beta = rs_new / rs_old;

    // 8. p = z + beta * p
    for (int i = 0; i < n_local; ++i) {
      p_local[i] = z_local[i] + beta * p_local[i];
    }

    // Update rs_old
    rs_old = rs_new;
  }
}