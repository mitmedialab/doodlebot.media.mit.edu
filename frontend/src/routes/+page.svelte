<script lang="ts">
    import UartService from "$lib/communication/UartService";

    const { bluetooth } = window.navigator;

    const constants = {
        handshakeMessage: "doodlebot",
        disconnectMessage: "disconnected",
        commandCompleteIdentifier: "done",
    };

    let popup: Window | null = null;

    let ip: string = localStorage.getItem("ip") || "";
    // Set fixed playground URL
    const playgroundURL: string = "http://doodlebot.media.mit.edu/playground";

    $: if (ip) localStorage.setItem("ip", ip);

    $: popupOrigin = popup ? new URL(playgroundURL).origin : "";

    let unsubscribers = new Array<() => void>();

    const unsubscribe = () => {
        for (const unsubscribe of unsubscribers) unsubscribe();
        unsubscribers = [];
    };

    const forwardToPopup = ({
        detail,
    }: Pick<CustomEvent<string>, "detail">) => {
        popup?.postMessage(detail, playgroundURL);
    };

    const disconnect = () =>
        forwardToPopup({ detail: constants.disconnectMessage });

    const getUartService = async (
        service: BluetoothRemoteGATTService,
        device: BluetoothDevice,
    ) => {
        const uartService = await UartService.create(service);

        uartService.addEventListener("receiveText", forwardToPopup);
        unsubscribers.push(() =>
            uartService.removeEventListener("receiveText", forwardToPopup),
        );

        device.addEventListener("gattserverdisconnected", () => {
            console.log("gattserverdisconnected");
            disconnect()
        });
        unsubscribers.push(() =>
            device.removeEventListener("gattserverdisconnected", disconnect),
        );

        return uartService;
    };

    const waitForPopupToRespond = () =>
        new Promise<void>((resolve) => {
            let interval = setInterval(
                () =>
                    popup!.postMessage(
                        constants.handshakeMessage,
                        playgroundURL,
                    ),
                100,
            );

            console.log("waiting for pop-up");

            const onReady = (event: MessageEvent) => {
                if (event.origin !== popupOrigin) {
                    return;
                }
                clearInterval(interval!);
                window.removeEventListener("message", onReady);
                resolve();
            };

            window.addEventListener("message", onReady);
        });

    const connect = async () => {
        if (!bluetooth) return alert("Bluetooth not supported");

        unsubscribe();

        const device = await bluetooth.requestDevice({
            filters: [{ services: [UartService.uuid] }],
        });

        if (!device) return alert("No device selected");
        if (!device.gatt) return alert("No GATT server");

        if (!device.gatt.connected) await device.gatt.connect();

        const services = await device.gatt.getPrimaryServices();

        const found = services.find(
            (service) => service.uuid === UartService.uuid,
        );

        if (!found) return alert("No UART service found");

        const uartService = await getUartService(found, device);

        // handle case if playgroundURL ends in question mark (?)
        const url = new URL(playgroundURL);
        url.searchParams.set("ip", ip);

        popup = window.open(url.toString());

        if (!popup) return alert("Please allow popups for this website");

        await waitForPopupToRespond();

        const forwardToBLE = async ({ data, origin }: MessageEvent) => {
            if (origin !== popupOrigin) {
                console.log("origin mismatch", origin, popupOrigin);
                return
            };
            console.log("sending data", data);
            await uartService.sendText(data);
            let text = data + constants.commandCompleteIdentifier;
            console.log("forwarding to popup", text);
            forwardToPopup({
                detail: data + constants.commandCompleteIdentifier,
            });
        };

        window.addEventListener("message", forwardToBLE);
        unsubscribers.push(() =>
            window.removeEventListener("message", forwardToBLE),
        );
    };
</script>

<div class="container">
    <div class="content">
        <h1>Welcome to a Day with Doodlebot</h1>
        <p class="subtitle">We will learn about AI and robotics</p>
        
        <div class="card">
            <div class="form-group">
                <label for="ip">Server IP Address:</label>
                <input id="ip" bind:value={ip} class="input" placeholder="Enter IP address" />
            </div>

            <button class="connect-button" on:click={connect}>Connect to Doodlebot</button>
        </div>
    </div>
</div>

<style>
    :global(body) {
        margin: 0;
        padding: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #FF9F1C, #FFBF69);
        color: #fff;
        height: 100vh;
        overflow: hidden;
    }
    
    .container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        padding: 0 2rem;
        box-sizing: border-box;
    }
    
    .content {
        text-align: center;
        max-width: 550px;
        width: 100%;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        margin-bottom: 2rem;
        color: #ffffff;
    }
    
    .card {
        background-color: rgba(255, 232, 214, 0.9); /* #FFE8D6 with opacity */
        border-radius: 12px;
        padding: 2rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .form-group {
        margin-bottom: 1.5rem;
        text-align: left;
    }
    
    label {
        display: block;
        margin-bottom: 0.5rem;
        font-weight: 600;
        font-size: 1rem;
        color: #CB997E; /* muted pink */
    }
    
    .input {
        width: 100%;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        border: 2px solid #FFBF69; /* light orange */
        background-color: white;
        color: #333;
        font-size: 1rem;
        box-sizing: border-box;
        transition: all 0.3s ease;
    }
    
    .input:focus {
        outline: none;
        border-color: #FF9F1C; /* bright orange */
        box-shadow: 0 0 0 3px rgba(255, 159, 28, 0.2);
    }
    
    .input::placeholder {
        color: rgba(203, 153, 126, 0.6); /* muted pink with opacity */
    }
    
    .connect-button {
        width: 100%;
        padding: 0.8rem;
        background-color: #FF9F1C; /* bright orange */
        color: white;
        border: none;
        border-radius: 6px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .connect-button:hover {
        background-color: #FFBF69; /* light orange */
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    .connect-button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
</style>
