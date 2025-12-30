{
  description = "A very basic flake";

  inputs = {
    nixpkgs = {
      type = "github";
      owner = "NixOS";
      repo = "nixpkgs";
      # nixpkgs-unstable:
      rev = "1b5c1881789eb8c86c655caeff3c918fb76fbfe6";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
          pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = false;
          };
        };
      in {
        devShells = {
          default = pkgs.mkShell {
            packages =
              with pkgs; [
                pnpm
		git
              ];
          };
        };
      }
    );

}
