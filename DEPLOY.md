# Deployment

The doodlebot application is currently deployed on an Ubuntu virtual machine issued to us by Necsys: [doodlebot.media.mit.edu](https://doodlebot.media.mit.edu)

It works by executing a series of [docker containers](https://www.docker.com/resources/what-container/):

- **backend:** This container executes the FastAPI application defined in [main.py](./main.py). The FastAPI is also responsible for serving the static assets of the **frontend** and **playground** containers described below.
- **frontend:** This container builds the [SvelteKit](https://svelte.dev/) web app defined in [frontend/](./frontend/) into a static website, which acts as the frontend you see when navigating to [doodlebot.media.mit.edu](https://doodlebot.media.mit.edu)
- **playground:** This container pulls down a single deployed folder of the [RAISE Playground repo](https://github.com/mitmedialab/prg-raise-playground)
    - Currently, it pulls down the deploy of the [doodlebot branch](https://github.com/mitmedialab/prg-raise-playground/tree/doodlebot), which is configured in [docker/playground.Dockerfile](./docker/playground.Dockerfile)
- **caddy:** This container executes the [Caddy](https://caddyserver.com/) web server, which routes traffic to the **backend** container

The activity of these containers is coordinated using [docker compose](https://docs.docker.com/compose/) as configured in [docker/compose.yml](./docker/compose.yml).


## Deployment Steps

The following steps demonstrate how to re-deploy the backend after pushing up changes from your local development environment (you should **NOT** develop directly on this machine).

The steps make reference to shell scripts stored directly on the deployment machine (and not in this repo, though they make use of scripts stored inside of the [cli/](./cli/) directory). These exist just to make the deployment process easier, but they often are pretty simple and you should definitely check out their contents (e.g. `cat ./example.sh`) to get a clear picture of what's going on 

1. Log onto the VM: `ssh <user_id>@doodlebot.media.mit.edu`
    - Use the `user_id` @pmalacho provided to you as well as the password when prompted
2. Create a sudo session: `sudo su`
    - You'll again be prompted for your password
3. Change directory to home: `cd`
3. Pull the latest changes for this repo: `./pull.sh`
4. (Re)start the docker containers: `./start.sh`
    - **NOTE:** If you deem it necessary, you can explicitly stop all currently running containers first: `./stop.sh`
5. When you run the above `./start.sh` command, you should see `nohup: ignoring input and appending output to 'nohup.out'`
    - This message indicates that we are running a process (specifically [cli/prod.sh](./cli/prod.sh)) through the [nohup](https://www.digitalocean.com/community/tutorials/nohup-command-in-linux) utility. It stands for "No Hang Up", which is useful for us so that the deployment process continues even when we exit the terminal (aka "hang up"). 
    - This command won't exit until all of the docker containers are built. If you need to see the output of this build process, you can open another terminal, ssh onto the machine, start a sudo session (`sudo su`) and run `./log-build.sh` (which literally just prints the contents of the `'nohup.out'` file referenced above).
        - As you can see, it's sometimes helpful to have two terminals connected to the deployment machine, especially when debugging.
6. Once the `./start.sh` command exits, the server will be starting up. During this time the site ([doodlebot.media.mit.edu](https://doodlebot.media.mit.edu)) will not be reachable -- in this sense, we do **NOT** currently support [zero downtime deployments](https://www.pingidentity.com/en/resources/blog/post/what-is-zero-downtime-deployment.html).
    - To see the output of the server at runtime, run `./log-runtime.sh`. 
    - Once you see `doodlebot-backend     | Mounted frontend at /` in the output of `./log-runtime.sh`, the full site should be responsive. 