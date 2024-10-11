package cn.edu.hitsz.compiler.parser;

import cn.edu.hitsz.compiler.NotImplementedException;
import cn.edu.hitsz.compiler.lexer.Token;
import cn.edu.hitsz.compiler.lexer.TokenKind;
import cn.edu.hitsz.compiler.parser.table.*;
import cn.edu.hitsz.compiler.symtab.SymbolTable;

import java.util.ArrayList;
import java.util.List;

//TODO: 实验二: 实现 LR 语法分析驱动程序

/**
 * LR 语法分析驱动程序
 * <br>
 * 该程序接受词法单元串与 LR 分析表 (action 和 goto 表), 按表对词法单元流进行分析, 执行对应动作, 并在执行动作时通知各注册的观察者.
 * <br>
 * 你应当按照被挖空的方法的文档实现对应方法, 你可以随意为该类添加你需要的私有成员对象, 但不应该再为此类添加公有接口, 也不应该改动未被挖空的方法,
 * 除非你已经同助教充分沟通, 并能证明你的修改的合理性, 且令助教确定可能被改动的评测方法. 随意修改该类的其它部分有可能导致自动评测出错而被扣分.
 */
public class SyntaxAnalyzer {
    private final SymbolTable symbolTable;
    private final List<ActionObserver> observers = new ArrayList<>();


    public SyntaxAnalyzer(SymbolTable symbolTable) {
        this.symbolTable = symbolTable;
    }

    /**
     * 注册新的观察者
     *
     * @param observer 观察者
     */
    public void registerObserver(ActionObserver observer) {
        observers.add(observer);
        observer.setSymbolTable(symbolTable);
    }

    /**
     * 在执行 shift 动作时通知各个观察者
     *
     * @param currentStatus 当前状态
     * @param currentToken  当前词法单元
     */
    public void callWhenInShift(Status currentStatus, Token currentToken) {
        for (final var listener : observers) {
            listener.whenShift(currentStatus, currentToken);
        }
    }

    /**
     * 在执行 reduce 动作时通知各个观察者
     *
     * @param currentStatus 当前状态
     * @param production    待规约的产生式
     */
    public void callWhenInReduce(Status currentStatus, Production production) {
        for (final var listener : observers) {
            listener.whenReduce(currentStatus, production);
        }
    }

    /**
     * 在执行 accept 动作时通知各个观察者
     *
     * @param currentStatus 当前状态
     */
    public void callWhenInAccept(Status currentStatus) {
        for (final var listener : observers) {
            listener.whenAccept(currentStatus);
        }
    }
    private List<Token> tokenList = new ArrayList<>();
    private int currentTokenIndex = 0;

    public void loadTokens(Iterable<Token> tokens) {
        // TODO: 加载词法单元
        // 你可以自行选择要如何存储词法单元, 譬如使用迭代器, 或是栈, 或是干脆使用一个 list 全存起来
        // 需要注意的是, 在实现驱动程序的过程中, 你会需要面对只读取一个 token 而不能消耗它的情况,
        // 在自行设计的时候请加以考虑此种情况
        for (Token token : tokens) {
            tokenList.add(token);
        }
    }


    private LRTable lrTable;
    private Status initStatus;

    public void loadLRTable(LRTable table) {
        // TODO: 加载 LR 分析表
        // 你可以自行选择要如何使用该表格:
        // 是直接对 LRTable 调用 getAction/getGoto, 抑或是直接将 initStatus 存起来使用
        this.lrTable = table;
        this.initStatus = table.getInit();
    }


    private Token getHead(Production production) {
        NonTerminal head = production.head();
        // 使用 simple 方法创建一个简单的 Token 对象
        return Token.simple(head.toString());
    }
    public void run() {
        // 初始化状态栈
        List<Status> stateStack = new ArrayList<>();
        stateStack.add(lrTable.getInit()); // 使用初始状态
        // 初始化符号栈
        List<Symbol> symbolStack = new ArrayList<>();
        symbolStack.add(new Symbol(Token.eof())); // 使用 EOF 符号初始化


        while (currentTokenIndex < tokenList.size()) {
            Token currentToken = tokenList.get(currentTokenIndex);
            Status currentStatus = stateStack.get(stateStack.size() - 1);
            Action action = lrTable.getAction(currentStatus, currentToken);

            switch (action.getKind()) {
                case Shift -> {
                    // 记录 Shift 操作
                    callWhenInShift(currentStatus, currentToken);

                    // 执行 Shift 操作
                    Status nextStatus = action.getStatus();
                    stateStack.add(nextStatus);
                    symbolStack.add(new Symbol(currentToken));

                    // 移动到下一个 Token
                    currentTokenIndex++;
                }
                case Reduce -> {
                    // 获取当前产生式
                    Production production = action.getProduction();

                    // 记录 Reduce 操作
                    callWhenInReduce(currentStatus, production);

                    // 从栈中弹出产生式右部的符号
                    for (int i = 0; i < production.body().size(); i++) {
                        stateStack.remove(stateStack.size() - 1);
                        symbolStack.remove(symbolStack.size() - 1);
                    }

                    // 获取当前状态栈顶状态
                    Status topStatus = stateStack.get(stateStack.size() - 1);

                    // 转移状态
                    Status gotoStatus = lrTable.getGoto(topStatus, production.head());
                    stateStack.add(gotoStatus);

                    // 将产生式左部符号压入符号栈
                    symbolStack.add(new Symbol(production.head()));
                }
                case Accept -> {
                    // 记录 Accept 操作
                    callWhenInAccept(currentStatus);
                    return;

                }
                case Error -> {
                    // 记录并处理错误
                    Action.error();
                    throw new RuntimeException("Syntax error at token: " + currentToken);
                }
            }
        }
    }

}
