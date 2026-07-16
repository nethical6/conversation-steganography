package main

import (
	"bufio"
	"encoding/base64"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

const usage = `decalgo - secure generative conversation codec

Usage:
  decalgo demo   [-conversation ID]
  decalgo encode [-conversation ID]
  decalgo decode [-conversation ID]
  decalgo generate -model MODEL [-prompt TEXT] [-top-n N] < payload > text
  decalgo extract  -model MODEL [-prompt TEXT] [-top-n N] < text > payload
  decalgo chain-send    -from NAME [-state FILE] < plaintext > record.json
  decalgo chain-receive [-state FILE] < record.json > plaintext
  decalgo chain-show    [-state FILE]
  decalgo chain-chat    -as NAME [-state FILE]
  decalgo chat          -conversation NAME -me NAME
  decalgo conversations

Set DECALGO_KEY to a base64-encoded key of at least 16 bytes. Generate one with:
  openssl rand -base64 32
For chat mode, two people can instead enter the same long shared phrase or set
DECALGO_SECRET. The phrase is not stored.

Modes:
  demo    Type plaintext; see the marked wire message and decoded plaintext.
  encode  Type plaintext; emit one marked wire message per line.
  decode  Paste one marked wire message per line; emit authenticated plaintext.
  generate  Encode stdin bytes into deterministic model token choices.
  extract   Recover stdin bytes from deterministic model token choices.
  chain-send     Add one sender's encrypted carrier to a persistent group chain.
  chain-receive  Authenticate, decrypt, and append the next group-chain record.
  chain-show     Print the locally known from/decrypted/encrypted conversation.
  chain-chat     Interactive multi-person tester with a warm model and state.
  chat           General messaging-app copy/paste client using a shared phrase.
  conversations  List locally stored named conversation states.

Enter /quit or press Ctrl-D to leave interactive modes. Chat state is encrypted
and persisted; messages must still be processed in their exact platform order.`

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(args []string, in io.Reader, out, errOut io.Writer) error {
	if len(args) == 0 || args[0] == "help" || args[0] == "-h" || args[0] == "--help" {
		fmt.Fprintln(out, usage)
		return nil
	}

	mode := args[0]
	if mode == "conversations" {
		if len(args) != 1 {
			return errors.New("conversations takes no arguments")
		}
		return listConversations(out)
	}
	if mode == "generate" || mode == "extract" {
		return runGenerative(mode, args[1:], in, out, errOut)
	}
	if mode == "chain-send" || mode == "chain-receive" || mode == "chain-show" || mode == "chain-chat" || mode == "chat" {
		return runChain(mode, args[1:], in, out, errOut)
	}
	if mode != "demo" && mode != "encode" && mode != "decode" {
		return fmt.Errorf("unknown mode %q\n\n%s", mode, usage)
	}
	fs := flag.NewFlagSet(mode, flag.ContinueOnError)
	fs.SetOutput(errOut)
	conversation := fs.String("conversation", "test-chat", "conversation identifier")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	if fs.NArg() != 0 {
		return errors.New("unexpected positional arguments")
	}

	keyText := strings.TrimSpace(os.Getenv("DECALGO_KEY"))
	if keyText == "" {
		return errors.New("DECALGO_KEY is not set; use a base64-encoded random key")
	}
	key, err := base64.StdEncoding.DecodeString(keyText)
	if err != nil {
		return errors.New("DECALGO_KEY must be standard base64")
	}
	if len(key) < 16 {
		return errors.New("DECALGO_KEY must decode to at least 16 bytes")
	}

	scanner := bufio.NewScanner(in)
	// Chat messages can be substantially larger than Scanner's default limit.
	scanner.Buffer(make([]byte, 4096), 1024*1024)

	switch mode {
	case "demo":
		return demo(scanner, out, key, *conversation)
	case "encode":
		return encode(scanner, out, key, *conversation)
	default:
		return decode(scanner, out, errOut, key, *conversation)
	}
}
