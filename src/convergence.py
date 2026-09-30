"""Validation-motivated, pre-test budget sensitivity audit; not a main-curve replacement."""
import subprocess
import sys


def main():
    for model in ['scratch','pretrained']:
        subprocess.run([sys.executable,'-m','src.train','--config',f'configs/convergence_{model}.yaml',
                        '--fraction','0.01','--seed','42','--output-root','runs/convergence'],check=True)


if __name__ == '__main__':
    main()
