const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;
let isInitialized = false;
let initializationPromise = null;

const mainPyPath = path.join(__dirname, '..', 'main.py');
console.log('Python script path:', mainPyPath);

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 700,
        height: 500,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        },
        titleBarStyle: 'hiddenInset',
        backgroundColor: '#00000000',
        transparent: true,
        opacity: 0.75,
        resizable: true
    });

    mainWindow.loadFile('index.html');
    
    if (process.env.NODE_ENV === 'development') {
        mainWindow.webContents.openDevTools();
    }
}

function initPythonProcess() {
    return new Promise((resolve, reject) => {
        if (pythonProcess) {
            console.log('Killing existing Python process');
            pythonProcess.kill();
        }

        console.log('Starting Python process with script:', mainPyPath);
        pythonProcess = spawn('python3', [mainPyPath], {
            stdio: ['pipe', 'pipe', 'pipe'],
            env: {
                ...process.env,
                PYTHONUNBUFFERED: '1',  // Ensure Python output is not buffered
                PYTHONPATH: path.join(__dirname, '..')  // Add the project root to Python path
            }
        });

        let initTimeout = setTimeout(() => {
            if (pythonProcess) {
                console.log('Python process initialization timed out');
                pythonProcess.kill();
            }
            reject(new Error('Python process initialization timeout'));
        }, 30000);

        pythonProcess.stdout.on('data', (data) => {
            const message = data.toString().trim();
            console.log('Python stdout:', message);
            if (message.startsWith('INIT:')) {
                console.log('Python process:', message);
                if (message.includes('ready')) {
                    isInitialized = true;
                    clearTimeout(initTimeout);
                    resolve();
                }
            } else if (isInitialized && !message.startsWith('DEBUG:')) {
                if (mainWindow && !mainWindow.isDestroyed()) {
                    mainWindow.webContents.send('python-response', message);
                }
            }
        });

        pythonProcess.stderr.on('data', (data) => {
            const message = data.toString().trim();
            console.log('Python stderr:', message);
            if (message.startsWith('DEBUG:')) {
                console.log('Python debug:', message);
            } else if (message.startsWith('ERROR:')) {
                console.error('Python error:', message);
                if (mainWindow && !mainWindow.isDestroyed()) {
                    mainWindow.webContents.send('python-error', message);
                }
            }
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python process exited with code ${code}`);
            isInitialized = false;
            initializationPromise = null;
            if (code !== 0) {
                console.log('Python process crashed, attempting to restart...');
                setTimeout(() => {
                    initPythonProcess().catch(console.error);
                }, 1000);
            }
        });

        pythonProcess.on('error', (err) => {
            console.error('Failed to start Python process:', err);
            clearTimeout(initTimeout);
            reject(err);
        });
    });
}

app.whenReady().then(() => {
    createWindow();
    initPythonProcess().catch(err => {
        console.error('Failed to initialize Python process:', err);
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

// handle IPC messages
ipcMain.handle('send-message', async (event, message) => {
    if (!isInitialized || !pythonProcess) {
        console.log('Waiting for Python process to initialize...');
        try {
            await initializationPromise;
        } catch (err) {
            console.error('Failed to wait for Python process initialization:', err);
            throw new Error('Python process initialization failed');
        }
    }

    return new Promise((resolve, reject) => {
        try {
            pythonProcess.stdin.write(message + '\n', (err) => {
                if (err) {
                    console.error('Error writing to Python process:', err);
                    reject(err);
                }
            });
        } catch (err) {
            console.error('Error in send-message handler:', err);
            reject(err);
        }
    });
}); 