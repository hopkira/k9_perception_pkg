def load_face_database(self):

    database = {}

    for person_dir in sorted(
        self.face_database.iterdir()
    ):

        if not person_dir.is_dir():
            continue

        embeddings = []

        for filename in sorted(
            person_dir.glob("*.npy")
        ):

            embedding = np.load(
                filename
            ).astype(
                np.float32
            ).flatten()

            norm = np.linalg.norm(
                embedding
            )

            if norm <= 0.0:
                continue

            embeddings.append(
                embedding / norm
            )

        if embeddings:
            database[
                person_dir.name
            ] = embeddings

            self.get_logger().info(
                f"Loaded {len(embeddings)} "
                f"embeddings for "
                f"'{person_dir.name}'"
            )

    return database