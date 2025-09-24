obk-build

colcon build --symlink-install --parallel-workers $(nproc)

if [ $? -eq 0 ]; then
    echo "✓ Build completed successfully!"
else
    echo "✗ Build failed with errors"
    exit 1
fi

source install/setup.bash

