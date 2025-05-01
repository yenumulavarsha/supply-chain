{ pkgs }:

let
  pythonEnv = pkgs.python311.withPackages (ps: with ps; [
    fastapi
    uvicorn
    altair
    pandas
    numpy
    plotly
    pydantic
    python-dotenv
    requests
    seaborn
    streamlit
    pip
  ]);
in {
  deps = [ pythonEnv ];
}
