{
  description = "Nova Hackathon frontend dev environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.nodejs_20
            pkgs.pnpm
            pkgs.git
            pkgs.python312Full
            pkgs.uv
          ];

          shellHook = ''
            export PATH=$PWD/node_modules/.bin:$PATH
            export NEXT_TELEMETRY_DISABLED=1
            
            # Use uv to create and activate a Python venv and install common dev packages on first enter
            if [ ! -d .venv ]; then
              echo "Creating Python virtualenv at .venv and installing packages: fastapi, numpy, torch, modal using uv"
              
              # Create venv using uv
              # The --python flag ensures uv uses the python312Full from the Nix environment
              uv venv .venv --python "${pkgs.python312Full}/bin/python"
            fi

            # Activate venv for interactive shell
            if [ -f .venv/bin/activate ]; then
              # shellcheck disable=SC1091
              source .venv/bin/activate
              
              # Use uv to install packages. The --upgrade-package ensures they are updated 
              # if they already exist, mimicking the previous pip install --upgrade
              uv pip install \
                fastapi \
                python-multipart \
                numpy \
                torch==2.8.0 \
                modal || true
            fi
          '';
        };
      });
}