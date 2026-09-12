/*
 * Fcitx5 ASR Native Bridge Addon
 * Minimal, crash-isolated Unix Domain Socket bridge for speech recognition text injection.
 */

#include <fcitx/addonfactory.h>
#include <fcitx/addoninstance.h>
#include <fcitx/addonmanager.h>
#include <fcitx/event.h>
#include <fcitx/inputcontext.h>
#include <fcitx/inputpanel.h>
#include <fcitx/instance.h>
#include <fcitx/text.h>
#include <fcitx-config/configuration.h>
#include <fcitx-config/iniparser.h>
#include <fcitx-config/option.h>
#include <fcitx-config/rawconfig.h>
#include <fcitx-utils/event.h>
#include <fcitx-utils/handlertable.h>
#include <fcitx-utils/key.h>
#include <fcitx-utils/keysym.h>
#include <fcitx-utils/log.h>
#include <fcitx-utils/fs.h>
#include <fcitx-utils/stringutils.h>
#include <fcitx-utils/textformatflags.h>

#include <fcntl.h>
#include <spawn.h>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

extern char **environ;

#include <cerrno>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fcitx {

FCITX_CONFIGURATION(
    AsrConfig,
    KeyListOption pttKeys{
        this, "PttKeys", "Push-to-talk keys",
        {Key("Control_R"), Key("Pause")},
        KeyListConstrain({KeyConstrainFlag::AllowModifierLess,
                          KeyConstrainFlag::AllowModifierOnly})};
);

class AsrAddon;

class ClientSession {
public:
    ClientSession(AsrAddon *addon, int fd) : addon_(addon), fd_(fd) {}
    ~ClientSession() {
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    void setSource(std::unique_ptr<EventSourceIO> source) {
        source_ = std::move(source);
    }

    bool onReadable();

private:
    AsrAddon *addon_;
    int fd_;
    std::unique_ptr<EventSourceIO> source_;
    std::string readBuf_;
};

class AsrAddon : public AddonInstance {
public:
    explicit AsrAddon(Instance *instance);
    ~AsrAddon() override;

    void reloadConfig() override;
    const Configuration *getConfig() const override { return &config_; }
    void setConfig(const RawConfig &config) override;

    void commitText(const std::string &text);
    void setPreedit(const std::string &text);
    void clearPreedit();
    std::string handlePttKeys(std::string_view args);

    void removeClient(int fd) {
        clients_.erase(fd);
    }

    Instance *instance() const { return instance_; }

private:
    void initSocket();
    void spawnDaemon();
    void watchDaemon();
    InputContext *targetIc() const;
    void handleNewConnection();
    void onKeyEvent(KeyEvent &event);
    bool sendPtt(bool down);
    bool isPttKey(const Key &raw) const;

    static constexpr char ConfPath[] = "conf/fcitx5-asr.conf";

    Instance *instance_;
    AsrConfig config_;
    int serverFd_ = -1;
    std::string socketPath_;
    std::unique_ptr<EventSourceIO> serverSource_;
    std::unique_ptr<EventSourceTime> daemonWatch_;
    std::unique_ptr<HandlerTableEntry<EventHandler>> keyWatch_;
    std::unordered_map<int, std::unique_ptr<ClientSession>> clients_;
};

bool ClientSession::onReadable() {
    char buf[1024];
    bool open = true;
    while (true) {
        ssize_t n = read(fd_, buf, sizeof(buf));
        if (n > 0) {
            readBuf_.append(buf, n);
        } else if (n == 0) {
            open = false;
            break;
        } else {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            open = false;
            break;
        }
    }

    size_t pos;
    while ((pos = readBuf_.find('\n')) != std::string::npos) {
        std::string line = readBuf_.substr(0, pos);
        readBuf_.erase(0, pos + 1);

        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.rfind("COMMIT ", 0) == 0) {
            FCITX_INFO() << "fcitx5-asr: COMMIT request: " << line.substr(7);
            addon_->commitText(line.substr(7));
            (void)write(fd_, "OK\n", 3);
        } else if (line.rfind("PREEDIT ", 0) == 0) {
            addon_->setPreedit(line.substr(8));
            (void)write(fd_, "OK\n", 3);
        } else if (line == "CLEAR") {
            addon_->clearPreedit();
            (void)write(fd_, "OK\n", 3);
        } else if (line == "PING") {
            const char pong[] = "PONG\n";
            (void)write(fd_, pong, sizeof(pong) - 1);
        } else if (line == "PTTKEYS" || line.rfind("PTTKEYS ", 0) == 0) {
            std::string args;
            if (line.size() > 7) {
                args = line.substr(7);
            }
            std::string out = addon_->handlePttKeys(args);
            out.push_back('\n');
            (void)write(fd_, out.data(), out.size());
        } else {
            (void)write(fd_, "ERR unknown\n", 12);
        }
    }
    return open;
}

AsrAddon::AsrAddon(Instance *instance) : instance_(instance) {
    reloadConfig();
    initSocket();
    spawnDaemon();
    watchDaemon();
    keyWatch_ = instance_->watchEvent(
        EventType::InputContextKeyEvent, EventWatcherPhase::PreInputMethod,
        [this](Event &event) { onKeyEvent(static_cast<KeyEvent &>(event)); });
}

namespace {
bool pathIsThisSocket(int fd, const std::string &path) {
    if (fd < 0 || path.empty()) {
        return false;
    }
    struct stat fd_st {};
    struct stat path_st {};
    if (fstat(fd, &fd_st) != 0 || stat(path.c_str(), &path_st) != 0) {
        return false;
    }
    return fd_st.st_dev == path_st.st_dev && fd_st.st_ino == path_st.st_ino;
}

std::string findAsrBinary() {
    if (const char *env = getenv("FCITX5_ASR_BIN"); env && env[0] != '\0') {
        if (fs::isexe(env)) {
            return env;
        }
    }
    std::vector<std::string> cands;
    if (const char *home = getenv("HOME"); home && home[0] != '\0') {
        cands.push_back(std::string(home) + "/.local/bin/fcitx5-asr");
        cands.push_back(std::string(home) + "/.cargo/bin/fcitx5-asr");
        cands.push_back(std::string(home) + "/.cargo/target/release/fcitx5-asr");
    }
    cands.push_back("/usr/local/bin/fcitx5-asr");
    cands.push_back("/usr/bin/fcitx5-asr");
    for (const auto &p : cands) {
        if (fs::isexe(p)) {
            return p;
        }
    }
    return "fcitx5-asr";
}

std::string runtimeFile(const char *name) {
    if (const char *runtimeDir = getenv("XDG_RUNTIME_DIR");
        runtimeDir && runtimeDir[0] != '\0') {
        return std::string(runtimeDir) + "/" + name;
    }
    return std::string("/tmp/") + name + "-" + std::to_string(getuid());
}

// startProcess keeps stdin/stdout/stderr. A foreground `fcitx5 -r` then
// leaves listen writing to the tty (and holding the mic) after Ctrl-C.
void spawnListener(const std::string &bin) {
    const std::string log = runtimeFile("fcitx5-asr.log");
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attr;
    if (posix_spawn_file_actions_init(&actions) != 0) {
        FCITX_ERROR() << "fcitx5-asr: posix_spawn_file_actions_init failed";
        return;
    }
    posix_spawnattr_init(&attr);
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSID);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null",
                                     O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, log.c_str(),
                                     O_WRONLY | O_CREAT | O_APPEND, 0600);
    posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO, STDERR_FILENO);

    std::string arg0 = bin;
    std::string arg1 = "listen";
    std::string arg2 = "--int8";
    std::string arg3 = "--trigger";
    std::string arg4 = "ptt";
    char *argv[] = {arg0.data(), arg1.data(), arg2.data(), arg3.data(),
                    arg4.data(), nullptr};

    pid_t pid = 0;
    int rc = posix_spawn(&pid, bin.c_str(), &actions, &attr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    if (rc != 0) {
        FCITX_ERROR() << "fcitx5-asr: failed to spawn listener: " << strerror(rc);
        return;
    }
    FCITX_INFO() << "fcitx5-asr: started listener pid=" << pid
                 << " log=" << log;
}

bool listenRunning() {
    const std::string path = runtimeFile("fcitx5-asr.lock");
    int fd = open(path.c_str(), O_RDWR | O_CREAT, 0600);
    if (fd < 0) {
        return false;
    }
    int rc = flock(fd, LOCK_EX | LOCK_NB);
    if (rc == 0) {
        flock(fd, LOCK_UN);
        close(fd);
        return false;
    }
    close(fd);
    return true;
}
} // namespace

AsrAddon::~AsrAddon() {
    serverSource_.reset();
    clients_.clear();
    if (serverFd_ >= 0) {
        // fcitx5 -r overlaps old and new instances. Unlinking blindly here
        // deletes the successor's socket file while its listen fd stays open.
        if (pathIsThisSocket(serverFd_, socketPath_)) {
            unlink(socketPath_.c_str());
        }
        close(serverFd_);
        serverFd_ = -1;
    }
}

void AsrAddon::reloadConfig() {
    readAsIni(config_, ConfPath);
}

void AsrAddon::setConfig(const RawConfig &config) {
    config_.load(config, true);
    safeSaveAsIni(config_, ConfPath);
}

std::string AsrAddon::handlePttKeys(std::string_view args) {
    const auto names = stringutils::split(stringutils::trimView(args), " ",
                                          stringutils::SplitBehavior::SkipEmpty);
    if (names.empty()) {
        return "OK " + Key::keyListToString(*config_.pttKeys);
    }
    KeyList keys;
    keys.reserve(names.size());
    for (const auto &name : names) {
        Key key(name);
        if (!key.isValid()) {
            return "ERR invalid key: " + name;
        }
        keys.push_back(key);
    }
    if (!config_.pttKeys.setValue(std::move(keys))) {
        return "ERR invalid key list";
    }
    safeSaveAsIni(config_, ConfPath);
    return "OK";
}

void AsrAddon::spawnDaemon() {
    if (getenv("FCITX5_ASR_NO_DAEMON")) {
        FCITX_INFO() << "fcitx5-asr: FCITX5_ASR_NO_DAEMON set, not starting listener";
        return;
    }
    const std::string bin = findAsrBinary();
    FCITX_INFO() << "fcitx5-asr: starting listener " << bin;
    spawnListener(bin);
}

void AsrAddon::watchDaemon() {
    daemonWatch_ = instance_->eventLoop().addTimeEvent(
        CLOCK_MONOTONIC, now(CLOCK_MONOTONIC) + 2000000ULL, 1000000ULL,
        [this](EventSourceTime *src, uint64_t) {
            if (!listenRunning()) {
                FCITX_INFO() << "fcitx5-asr: listener died, restarting";
                spawnDaemon();
            }
            src->setTime(now(CLOCK_MONOTONIC) + 2000000ULL);
            return true;
        });
}

InputContext *AsrAddon::targetIc() const {
    auto *ic = instance_->inputContextManager().lastFocusedInputContext();
    if (!ic) {
        ic = instance_->mostRecentInputContext();
    }
    return ic;
}

void AsrAddon::initSocket() {
    const char *runtimeDir = getenv("XDG_RUNTIME_DIR");
    if (runtimeDir && runtimeDir[0] != '\0') {
        socketPath_ = std::string(runtimeDir) + "/fcitx5-asr.sock";
    } else {
        socketPath_ = "/tmp/fcitx5-asr-" + std::to_string(getuid()) + ".sock";
    }

    unlink(socketPath_.c_str());

    serverFd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    if (serverFd_ < 0) {
        FCITX_ERROR() << "fcitx5-asr: Failed to create socket: " << strerror(errno);
        return;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socketPath_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(serverFd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        FCITX_ERROR() << "fcitx5-asr: Failed to bind socket " << socketPath_ << ": " << strerror(errno);
        close(serverFd_);
        serverFd_ = -1;
        return;
    }

    chmod(socketPath_.c_str(), 0600);

    if (listen(serverFd_, 8) < 0) {
        FCITX_ERROR() << "fcitx5-asr: Failed to listen on socket: " << strerror(errno);
        close(serverFd_);
        serverFd_ = -1;
        return;
    }

    serverSource_ = instance_->eventLoop().addIOEvent(
        serverFd_, IOEventFlag::In,
        [this](EventSourceIO *, int, IOEventFlags) {
            handleNewConnection();
            return true;
        });

    FCITX_INFO() << "fcitx5-asr: Listening on " << socketPath_;
}

bool AsrAddon::isPttKey(const Key &raw) const {
    const auto sym = raw.sym();
    for (const auto &key : *config_.pttKeys) {
        if (sym == key.sym()) {
            return true;
        }
    }
    return false;
}

void AsrAddon::onKeyEvent(KeyEvent &event) {
    if (!isPttKey(event.rawKey())) {
        return;
    }
    if (sendPtt(!event.isRelease())) {
        event.filterAndAccept();
    }
}

bool AsrAddon::sendPtt(bool down) {
    std::string path;
    const char *runtimeDir = getenv("XDG_RUNTIME_DIR");
    if (runtimeDir && runtimeDir[0] != '\0') {
        path = std::string(runtimeDir) + "/fcitx5-asr-ctl.sock";
    } else {
        path = "/tmp/fcitx5-asr-ctl-" + std::to_string(getuid()) + ".sock";
    }

    int fd = socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        return false;
    }
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);
    const char *msg = down ? "DOWN" : "UP";
    ssize_t n = sendto(fd, msg, strlen(msg), MSG_DONTWAIT,
                       reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr));
    close(fd);
    return n >= 0;
}

void AsrAddon::handleNewConnection() {
    while (true) {
        int clientFd = accept4(serverFd_, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
        if (clientFd < 0) {
            break;
        }

        auto session = std::make_unique<ClientSession>(this, clientFd);
        auto *sessionPtr = session.get();

        auto source = instance_->eventLoop().addIOEvent(
            clientFd, IOEventFlag::In,
            [this, sessionPtr, clientFd](EventSourceIO *, int, IOEventFlags) {
                if (!sessionPtr->onReadable()) {
                    removeClient(clientFd);
                    return false;
                }
                return true;
            });

        session->setSource(std::move(source));
        clients_[clientFd] = std::move(session);
    }
}

void AsrAddon::commitText(const std::string &text) {
    if (text.empty()) {
        return;
    }
    auto *ic = targetIc();
    if (!ic) {
        FCITX_WARN() << "fcitx5-asr: COMMIT dropped: no recent input context";
        return;
    }
    FCITX_INFO() << "fcitx5-asr: committing to input context";
    ic->inputPanel().reset();
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
    ic->commitString(text);
    instance_->flushUI();
}

void AsrAddon::setPreedit(const std::string &text) {
    auto *ic = targetIc();
    if (!ic) {
        return;
    }
    ic->inputPanel().reset();
    if (!text.empty()) {
        // Floating panel only. Client preedit would type "识别中..." into the app.
        ic->inputPanel().setAuxUp(Text("ASR"));
        ic->inputPanel().setPreedit(Text(text, TextFormatFlag::Underline));
    }
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
    instance_->flushUI();
}

void AsrAddon::clearPreedit() {
    auto *ic = targetIc();
    if (!ic) {
        return;
    }
    ic->inputPanel().reset();
    ic->updatePreedit();
    ic->updateUserInterface(UserInterfaceComponent::InputPanel);
    instance_->flushUI();
}

class AsrAddonFactory : public AddonFactory {
public:
    AddonInstance *create(AddonManager *manager) override {
        return new AsrAddon(manager->instance());
    }
};

} // namespace fcitx

FCITX_ADDON_FACTORY(fcitx::AsrAddonFactory)
