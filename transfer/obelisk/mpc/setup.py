import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'mpc'

def package_files(base_dir: str, package_name: str, patterns=["**/*"]):
    """Recursively collect all files matching patterns under base_dir."""
    paths = []
    for pattern in patterns:
        for path in glob(os.path.join(base_dir, pattern), recursive=True):
            if os.path.isfile(path):
                # Create destination path under share/package_name/...
                rel_path = os.path.relpath(path, base_dir)
                install_path = os.path.join("share", package_name, base_dir, os.path.dirname(rel_path))
                paths.append((install_path, [path]))
    return paths


data_files = [
    ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
    ("share/" + package_name, ["package.xml"]),
    (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yml"))),
    (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.obk"))),
    (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.py"))),
    (os.path.join("share", package_name, "rviz"), glob(os.path.join("rviz", "*.rviz"))),
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='zolkin',
    maintainer_email='zach.olkin@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "mpc_node = mpc.mpc_node:main",
        ],
    },
)
