import trimesh

if __name__ == '__main__':
    face = trimesh.load('face_model_with_iris.obj')
    assert type(face) == trimesh.Trimesh, type(face)
    print(face.vertices)
