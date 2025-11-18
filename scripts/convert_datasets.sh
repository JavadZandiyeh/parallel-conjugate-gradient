if [ -z "$PETSC_DIR" ]; then
  echo "Error: PETSC_DIR is not set" >&2
  exit 1
fi

while IFS= read -r dataset; do
  python3 $PETSC_DIR/lib/petsc/bin/PetscBinaryIO.py convert "$dataset"
done < datasets/matrices_list.txt