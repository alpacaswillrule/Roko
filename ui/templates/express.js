import express from 'express';
import cors from 'cors';  
import {
    Client,
    PrivateKey,
    AccountId,
    TransferTransaction,
    Hbar,
    HbarUnit,
    AccountBalanceQuery,
} from '@hashgraph/sdk';
import dotenv from 'dotenv';
// import fetch from 'node-fetch';

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

const port = 3000;
let client;

app.post('/transferHbar', async (req, res) => {
    try {
        console.log("hwre", req.body)
        console.log()
        const {cost} = req.body;

        const operatorIdStr = process.env.OPERATOR_ACCOUNT_ID;
        const operatorKeyStr = process.env.OPERATOR_ACCOUNT_PRIVATE_KEY;
        const minerIdStr = process.env.ACCOUNT_0_ID

        if (!operatorIdStr || !operatorKeyStr || ! minerIdStr) {
            throw new Error('Missing OPERATOR_ACCOUNT_ID or OPERATOR_ACCOUNT_PRIVATE_KEY');
        }

        const operatorId = AccountId.fromString(operatorIdStr);
        const minerId = AccountId.fromString(minerIdStr);
        const operatorKey = PrivateKey.fromStringECDSA(operatorKeyStr);

        client = Client.forTestnet().setOperator(operatorId, operatorKey);
        client.setDefaultMaxTransactionFee(new Hbar(100));
        client.setDefaultMaxQueryPayment(new Hbar(50));

        const currentDateTime = new Date().toLocaleString(); // Get current date and time
        const memoMessage = `Query transaction at Roko at ${currentDateTime}`;
        
        // Convert cost to integer tinybars and ensure splits are exact
        const costInTinybars = parseInt(cost * 100000000); // Convert to tinybars as integer
        const minerAmount = parseInt(costInTinybars * 0.4); // 40% to miner
        const otherAmount = costInTinybars - minerAmount; // Remainder to ensure exact total

        const transferTx = await new TransferTransaction()
            .setTransactionMemo(memoMessage)
            .addHbarTransfer(operatorId, new Hbar(-costInTinybars, HbarUnit.Tinybar))
            .addHbarTransfer(minerId, new Hbar(minerAmount, HbarUnit.Tinybar))
            .addHbarTransfer('0.0.5615771', new Hbar(otherAmount, HbarUnit.Tinybar))
            .freezeWith(client);


        console.log("here")

        const transferTxSigned = await transferTx.sign(operatorKey);
        const transferTxSubmitted = await transferTxSigned.execute(client);
        const transferTxReceipt = await transferTxSubmitted.getReceipt(client);
        const transactionStatus = transferTxReceipt.status;

        // 🔹 Get Account Balance
        const newAccountBalance = await new AccountBalanceQuery()
            .setAccountId(operatorId)
            .execute(client);

        const newHbarBalance = newAccountBalance.hbars;

        // 🔹 Transaction Verification URL
        const transferTxId = transferTx.transactionId;
        const transferTxVerifyUrl = `https://hashscan.io/testnet/transaction/${transferTxId}`;

        client.close();

        // Send response with transaction details
        return res.json({
            status: transactionStatus.toString(),
            transactionUrl: transferTxVerifyUrl,
        });

    } catch (ex) {
        if (client) {
            client.close();
        }
        // Only send error response if no response has been sent yet
        if (!res.headersSent) {
            return res.status(500).json({ error: ex.message });
        }
    }
});

app.get('/getUserBalance', async (req, res) => {
    try {
        const operatorIdStr = process.env.OPERATOR_ACCOUNT_ID;
        const operatorKeyStr = process.env.OPERATOR_ACCOUNT_PRIVATE_KEY;

        if (!operatorIdStr || !operatorKeyStr) {
            throw new Error('Missing ID or PK');
        }

        const operatorId = AccountId.fromString(operatorIdStr);
        const operatorKey = PrivateKey.fromStringECDSA(operatorKeyStr);

        client = Client.forTestnet().setOperator(operatorId, operatorKey);
        client.setDefaultMaxTransactionFee(new Hbar(100));
        client.setDefaultMaxQueryPayment(new Hbar(50));

        // 🔹 Get Account Balance
        const accountBalance = await new AccountBalanceQuery()
            .setAccountId(operatorId)
            .execute(client);

        const hbarBalance = accountBalance.hbars.toString();

        client.close();

        return res.json({
            status: 'success',
            balance: hbarBalance,
        });

    } catch (ex) {
        if (client) {
            client.close();
        }
        if (!res.headersSent) {
            return res.status(500).json({ error: ex.message });
        }
    }
});


app.get('/getMinerBalance', async (req, res) => {
    try {
        const operatorIdStr = process.env.ACCOUNT_0_ID;
        const operatorKeyStr = process.env.ACCOUNT_0_PRIVATE_KEY;

        if (!operatorIdStr || !operatorKeyStr) {
            throw new Error('Missing ID or PK');
        }

        const operatorId = AccountId.fromString(operatorIdStr);
        const operatorKey = PrivateKey.fromStringECDSA(operatorKeyStr);

        client = Client.forTestnet().setOperator(operatorId, operatorKey);
        client.setDefaultMaxTransactionFee(new Hbar(100));
        client.setDefaultMaxQueryPayment(new Hbar(50));

        // 🔹 Get Account Balance
        const accountBalance = await new AccountBalanceQuery()
            .setAccountId(operatorId)
            .execute(client);

        const hbarBalance = accountBalance.hbars.toString();

        client.close();

        return res.json({
            status: 'success',
            balance: hbarBalance,
        });

    } catch (ex) {
        if (client) {
            client.close();
        }
        if (!res.headersSent) {
            return res.status(500).json({ error: ex.message });
        }
    }
});


// Simple health check endpoint
app.get("/", (req, res) => {
    res.send("Plz work");
});

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});
