import os.path as osp
import subprocess
import sys
from collections import defaultdict
import os
import cv2
import torch

import mmcv


def collect_env():
    """Collect the information of the running environments.

    Returns:
        dict: The environment information. The following fields are contained.

            - sys.platform: The variable of ``sys.platform``.
            - Python: Python version.
            - CUDA available: Bool, indicating if CUDA is available.
            - GPU devices: Device type of each GPU.
            - CUDA_HOME (optional): The env var ``CUDA_HOME``.
            - NVCC (optional): NVCC version.
            - GCC: GCC version, "n/a" if GCC is not installed.
            - MSVC: Microsoft Virtual C++ Compiler version, Windows only.
            - PyTorch: PyTorch version.
            - PyTorch compiling details: The output of \
                ``torch.__config__.show()``.
            - TorchVision (optional): TorchVision version.
            - OpenCV: OpenCV version.
            - MMCV: MMCV version.
            - MMCV Compiler: The GCC version for compiling MMCV ops.
            - MMCV CUDA Compiler: The CUDA version for compiling MMCV ops.
    """
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        from mmcv.utils.parrots_wrapper import _get_cuda_home
        CUDA_HOME = _get_cuda_home()
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(f'"{nvcc}" -V', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
                release = nvcc.rfind('Cuda compilation tools')
                build = nvcc.rfind('Build ')
                nvcc = nvcc[release:build].strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    try:
        # Check C++ Compiler.
        # For Unix-like, sysconfig has 'CC' variable like 'gcc -pthread ...',
        # indicating the compiler used, we use this to get the compiler name
        import sysconfig
        cc = sysconfig.get_config_var('CC')
        if cc:
            cc = osp.basename(cc.split()[0])
            cc_info = subprocess.check_output(f'{cc} --version', shell=True)
            env_info['GCC'] = cc_info.decode('utf-8').partition(
                '\n')[0].strip()
        else:
            # on Windows, cl.exe is not in PATH. We need to find the path.
            # distutils.ccompiler.new_compiler() returns a msvccompiler
            # object and after initialization, path to cl.exe is found.
            import locale
            import os
            from distutils.ccompiler import new_compiler
            ccompiler = new_compiler()
            ccompiler.initialize()
            cc = subprocess.check_output(
                f'{ccompiler.cc}', stderr=subprocess.STDOUT, shell=True)
            encoding = os.device_encoding(
                sys.stdout.fileno()) or locale.getpreferredencoding()
            env_info['MSVC'] = cc.decode(encoding).partition('\n')[0].strip()
            env_info['GCC'] = 'n/a'
    except subprocess.CalledProcessError:
        env_info['GCC'] = 'n/a'

    env_info['PyTorch'] = torch.__version__

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass

    env_info['OpenCV'] = cv2.__version__

    env_info['MMCV'] = mmcv.__version__

    try:
        from mmcv.ops import get_compiler_version, get_compiling_cuda_version
    except ModuleNotFoundError:
        env_info['MMCV Compiler'] = 'n/a'
        env_info['MMCV CUDA Compiler'] = 'n/a'
    else:
        env_info['MMCV Compiler'] = get_compiler_version()
        env_info['MMCV CUDA Compiler'] = get_compiling_cuda_version()

    return env_info


def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
    return out


def get_git_hash(fallback='HASH_N.A.', digits=None):
    """Get the git hash of the current repo.

    Args:
        fallback (str, optional): The fallback string when git hash is
            unavailable. Defaults to 'unknown'.
        digits (int, optional): kept digits of the hash. Defaults to None,
            meaning all digits are kept.

    Returns:
        str: Git commit hash.
    """

    if digits is not None and not isinstance(digits, int):
        raise TypeError('digits must be None or an integer')

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
        if digits is not None:
            sha = sha[:digits]
    except OSError as e:
        print(f"ERROR: {e}")
        sha = fallback

    return sha

def get_git_branch():
    try:
        out = _minimal_ext_cmd(['git', 'branch', '--show-current'])
        branch = out.strip().decode('ascii')
    except Exception as e:
        print(f"ERROR: {e}")
        branch = "branch_N.A."
    
    return branch

def get_git_info():
    return f"branch={get_git_branch()}, hash={get_git_hash()}"